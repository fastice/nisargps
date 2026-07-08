#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:31:25 2024

@author: ian
"""
import functools
import numpy as np
import calendar
from datetime import timedelta, datetime
import os
import pyproj
from scipy.stats import linregress
import nisarcryodb
import sys


class nisarStation():
    '''
    Abstract class to define parser for NISAR HDF products.
    '''
    def catchError(func):
        ''' Decorator to trap and abbreviate errors '''
        @functools.wraps(func)
        def catchErrorInner(inst, *args, **kwargs):
            traceBack = False
            if 'traceBack' in kwargs:
                traceBack = kwargs['traceBack']
            try:
                return func(inst, *args, **kwargs)
            except Exception as errMsg:
                excType, excObj, excTb = sys.exc_info()
                tb = excTb
                while tb is not None:
                    line = tb.tb_lineno
                    tb = tb.tb_next
                msg = f'Error in: {type(inst).__name__}.{func.__name__} at ' \
                    f'line {line} \nMessage: {errMsg}'
                inst.printError(msg)

                def myExit(traceBack):
                    try:
                        sys.exit()
                    except SystemExit:
                        if traceBack:
                            sys.exit()
                # This will only do a full exit if traceBack is True
                myExit(traceBack)
        return catchErrorInner

    @catchError
    def __init__(self, stationName, epsg=None, useDB=True, DBConnection=None,
                 DBConfigFile='calvaldb_config.ini', cacheDB=True, debugFile=None,
                 **kwargs):
        '''
        Class for handling GPS data for a NISAR station

        Parameters
        ----------
        stationName : str
            Four character station ID.
        epsg : str, optional
            Epsg code for for Arctic (3413) or Antarctic (3031). Default is
            None to autodetect based on north or south latitudes.
        useDB : bool, optional
            Get GPS from database. Alternatively read from text file
            (not fully supported). Default is True.
        DBConnection: nisarcryodb
            Pass in a connection, which can be used for multiple stations.
            The default None, which opens a dedicated connection.
        DBConfigFile: str, optional
            Database config file path. Default is ./calvaldb_config.ini.
        traceBack: bool, optional
            By default error handler prints abreviated messages. Set true for
            full   traceback.
        **kwords : dict
            Keywords to pass to other methods (.e.g., error handler).

        Returns
        -------
        None.

        '''
        self.stationName = stationName
        # Variables to store text input data
        self.date = np.array([])
        self.epoch = np.array([])
        self.lat = np.array([])
        self.lon = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.sigma3 = np.array([])
        #
        self.epsg = epsg
        self.lltoxy = None
        self.eceftoll = None
        self.lltoecef = None
        self.cacheDB = cacheDB
        #
        if useDB and DBConnection is None:
            self.DB = nisarcryodb.nisarcryodb(configFile=DBConfigFile)
        elif DBConnection is not None:
            self.DB = DBConnection
        else:
            self.DB = None
        # Set station id if using a database
        self.stationID = None
        if self.DB is not None:
            self.stationID = self.DB.stationNameToID(stationName)
        print(self.stationID)
        #
        # cache db for faster access
        self.DBData = None
        if self.DB is not None and self.cacheDB:
            print('Caching station data')
            self.DBData = self.DB.getTableListing(schemaName='landice',
                                                  tableName='gps_data',
                                                  filters={'station_id': self.stationID})
 
        # Cache year lengths for speed
        self._computeYearLengthLookUp(**kwargs)
        # Add decimal_year derived from time_utc (new schema has no decimal_year column)
        if self.DBData is not None:
            self.DBData = self._addDecimalYear(self.DBData)
        # Debug mode: skip DB entirely and load cached data from parquet
        if debugFile is not None:
            import pandas as pd
            self.DBData = self._addDecimalYear(pd.read_parquet(debugFile))

    @catchError
    def _computeYearLengthLookUp(self, **kwargs):
        '''
        Cache year lengths to speedup date conversions
        '''
        years = range(1990, 2100)
        lengths = np.array(
            [(datetime(y + 1, 1, 1) - datetime(y, 1, 1)).total_seconds()
             for y in years])
        self.yearLengthLookupSeconds = dict(zip(years, lengths))
        self.yearLengthLookupDays = dict(zip(years, lengths/86400))

    @catchError
    def _addDecimalYear(self, data, **kwargs):
        '''
        Add a decimal_year column computed from time_utc.
        New schema removed decimal_year; this restores it so downstream
        code that reads data['decimal_year'] works unchanged.
        '''
        import pandas as pd
        dates = pd.to_datetime(data['time_utc'])
        data = data.copy()
        data['decimal_year'] = dates.apply(
            lambda d: self._datetimeToDecimalYear(d.to_pydatetime()))
        return data

    def printError(self, msg):
        '''
        Print error message.
        Parameters
        ----------
        msg : str
            error message.
        Returns
        -------
        None
        '''
        length = max([len(x) for x in msg.split('\n')])
        stars = ''.join(['*']*length)
        print(f'\n\033[1;31m{stars}\n{msg} \n{stars}\n \033[0m\n')

    @catchError
    def _initCoordinateConversion(self, **kwargs):
        '''
        Use pyproj to setup lltoxy conversion for epsg.
        Returns
        -------
        None.
        '''
        print('EPSG', self.epsg)
        self.crs = pyproj.CRS.from_epsg(str(self.epsg))
        self.proj = pyproj.Proj(str(self.epsg))
        if self.lltoxy is None:
            self.lltoxy = pyproj.Transformer.from_crs("EPSG:4326",
                                                      f"EPSG:{self.epsg}"
                                                      ).transform
        if self.eceftoll is None:
            self.eceftoll = pyproj.Transformer.from_crs("EPSG:4978",
                                                        "EPSG:4326"
                                                       ).transform
        if self.lltoecef is None:
            self.lltoecef = pyproj.Transformer.from_crs("EPSG:4326",
                                                        "EPSG:4978"
                                                       ).transform

    @catchError
    def _determineEPSG(self, lat, **kwargs):
        '''
        Determine epsg base on lat (3031, 3413) based on a lat value
        lat: float
            latitude value used to detect depsg.
        Returns
        -------
        None.

        '''
        if self.epsg is None:
            if lat < -55:
                self.epsg = 3031
            elif lat > 55:
                self.epsg = 3413
            else:
                self._printError('Mid-band latitude(<|55| deg), cannot '
                                 'autodetect epsg')
        #

    @catchError
    def _readFile(self, filePath, **kwargs):
        '''
        Read a JPL processed GPS text file for no DB case.

        Parameters
        ----------
        filePath : str
            Path to GPS file.

        Returns
        -------
        date, decDate, lat, lon, z, sigma

        '''
        if not os.path.exists(filePath):
            self.printError(f'Cannot open {filePath}')
        newData = []
        date = []
        count = 0
        with open(filePath) as fpGPS:
            for line in fpGPS:
                # Process line
                pieces = line.split()
                if len(pieces) != 6 or self.stationName != pieces[-1].strip():
                    print(f'skipping line {count} missing data or invalid '
                          'station')
                # Grab data
                lineData = [float(x) for x in pieces[0:-1]]
                # compute datetime
                year = int(lineData[0])
                sec = np.rint((lineData[0] - year) *
                              (365 + int(calendar.isleap(year))) * 86400)
                date.append(datetime(year, 1, 1, 0, 0, 0) +
                            timedelta(seconds=sec))
                newData.append(lineData)
                count += 1
        # returns date, epoch, lat, lon, z, sigmax
        epoch, lat, lon, z, sigma = np.transpose(newData)

        return np.array(date),  epoch, lat, lon, z, sigma

    @catchError
    def addData(self, filePath, **kwargs):
        '''
         Read data and merge with any existing data

        Parameters
        ----------
        filePath : str
            Path to GPS file.

        Returns
        -------
        None.

        '''
        date, epoch, lat, lon, z, sigma3 = self._readFile(filePath)
        #
        self._determineEPSG(lat[0])
        #
        x, y = self.lltoxy(lat, lon)
        # add data
        for var, data in zip(
                ['date', 'epoch',  'lat', 'lon', 'x', 'y', 'z', 'sigma3'],
                [date, epoch, lat, lon, x, y, z, sigma3]):
            setattr(self, var, np.append(data, getattr(self, var)))
        #

        # Now make sure all monotonic in time
        sortOrder = np.argsort(self.epoch)
        for var, data in zip(['date', 'epoch',  'lat', 'lon', 'x', 'y', 'z',
                              'sigma3'],
                             [date, epoch, lat, lon, x, y, z, sigma3]):
            setattr(self, var, getattr(self, var)[sortOrder])
        #
        self.meanLat = np.mean(lat)
        #
        self.projLengthScale = self.proj.get_factors(0, self.meanLat
                                                     ).parallel_scale

    @catchError
    def computeVelocity(self, date1, date2, method='regression', minPoints=10,
                        dateFormat='%Y-%m-%d', averagingPeriod=12, tides=True,
                        filters={}, **kwargs):
        '''
         Compute velocity for date range

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date.
        method : str, optional
            Use either point or regression. The default is 'regression'
        dateFormat : str optional
            Date format for data1 and date2. The default is '%Y-%m-%d'.
        minPoints : int, optional
            Return nan's if the number of valid points is < minPoits. The
            default is 10.
        averagingPeriod : int or float, optional
            For 'point' method only. The number of hours on either side of
            date1 & date2 to average when computing the two point positions.
            The default is 12 to yield the difference of two daily averages.
        Returns
        -------
        vvx, vy, x, y : velocity (vx,vy) and mean location (x, y) of the
        measurement.
        '''
        if method == 'regression':
            return self.computeVelocityRegression(date1, date2,
                                                  minPoints=minPoints,
                                                  dateFormat=dateFormat,
                                                  filters=filters, tides=tides,
                                                  **kwargs)
        elif method == 'point':
            return self.computeVelocityPtToPt(date1, date2,
                                              minPoints=minPoints,
                                              dateFormat=dateFormat,
                                              averagingPeriod=averagingPeriod,
                                              filters=filters, tides=tides,
                                              **kwargs)
        else:
            self.printError(
                f'Invalid method {method}, use point or regression')
            return np.nan, np.nan, np.nan, np.nan

    @catchError
    def computeVelocityRegression(self, date1, date2, minPoints=10,
                                  dateFormat='%Y-%m-%d', filters={},
                                  computeVz=False, tides=True,
                                  **kwargs):
        '''
         Compute velocity for date range

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date..
        dateFormat : str, optional
            Format for date1 and date2. The default is '%Y-%m-%d'.
        minPoints : int, optional
            Return nan's if the number of valid points is < minPoits. The
            default is 10.
        Returns
        -------
        vx, vy, vz, x, y : velocity (vx,vy) and mean location (x, y) of the
        measurement.
        '''
        date, x, y, z, epoch = self.subsetXYZ(date1, date2,
                                              dateFormat=dateFormat, tides=tides,
                                              filters=filters, **kwargs)
        if x is np.nan:
            if computeVz:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                return np.nan, np.nan, np.nan, np.nan
        #
        # Uses slope of linear regression as velocity estimate
        vxPS, intercept, rx, px, sigmax = linregress(epoch, x)
        vyPS, intercept, ry, py, sigmay = linregress(epoch, y)
        # Scale from projected to actual coordinates
        vxPS = vxPS/self.projLengthScale
        vyPS = vyPS/self.projLengthScale
        xMean = np.mean(x)
        yMean = np.mean(y)
        if not computeVz:
            return vxPS, vyPS, xMean, yMean
        vz, intercept, rx, px, sigmaz = linregress(epoch, z)
        print(f'vz={vz:.2f},$$r^2$$={rx*rx:.2f}, p={px:.3f}, sigma={sigmaz:.4f}')
        return vxPS, vyPS, vz, xMean, yMean
    
    @catchError
    def computeVelocityPtToPt(self, date1, date2, minPoints=10,
                              dateFormat='%Y-%m-%d', averagingPeriod=12,
                              computeVz=False, tides=True,
                              filters={},
                              **kwargs):
        '''
        Compute velocity for date range differencing point positions

        Parameters
        ----------
        date1 : datetime date
            First date in interval to compute date.
        date2 : datetime date
            Second date in interval to compute date.
        dateFormat : TYPE, optional
            Format for date1 and date2. The default is '%Y-%m-%d'. The default
            is '%Y-%m-%d'.
        minPoints : int, optional
            Return nan's if the number of valid points is < minPoits. The
            default is 10.
       averagingPeriod : int or float, optional
            The number of hours on either side of date1 & date2 to average
            when computing the two point positions. The default is 12 to
            yield the difference of two daily averages.

        Returns
        -------
        vx, vy, x, y : velocity (vx,vy) and mean location (x, y) of the
        measurement.
        '''
        date1 = self._formatDate(date1, dateFormat=dateFormat)
        date2 = self._formatDate(date2, dateFormat=dateFormat)
        tAvg = timedelta(hours=averagingPeriod)
        #
        dates1, x1, y1, z1, epoch1 = self.subsetXYZ(date1 - tAvg,
                                                    date1 + tAvg,
                                                    minPoints=minPoints,
                                                    filters=filters,tides=tides,
                                                    **kwargs)
        dates2, x2, y2, z2, epoch2 = self.subsetXYZ(date2 - tAvg,
                                                    date2 + tAvg,
                                                    minPoints=minPoints,
                                                    filters=filters, tides=tides,
                                                    **kwargs)
        if x1 is np.nan or x2 is np.nan:
            if computeVz:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                return np.nan, np.nan, np.nan, np.nan
        #
        # Compute averages centered on date1 and date2
        x1Avg, x2Avg = np.mean(x1), np.mean(x2)
        y1Avg, y2Avg = np.mean(y1), np.mean(y2)
        epoch1Avg, epoch2Avg = np.mean(epoch1), np.mean(epoch2)
        
        dT = epoch2Avg - epoch1Avg
        #
        vxPS = (x2Avg - x1Avg) / dT
        vyPS = (y2Avg - y1Avg) / dT
        xPos = (x1Avg + x2Avg) * 0.5
        yPos = (y1Avg + y2Avg) * 0.5
       
        if computeVz:
            z1Avg, z2Avg = np.mean(z1), np.mean(z2)
            vz = (z2Avg - z1Avg) / dT
            return vxPS/self.projLengthScale, vyPS/self.projLengthScale, vz, xPos, yPos
        else:
            # Scale from projected to actual coordinates
            return vxPS/self.projLengthScale, vyPS/self.projLengthScale, xPos, yPos
 

    @catchError
    def computeVelocityTimeSeries(self, date1, date2, dT, sampleInterval,
                                  method='regression', dateFormat='%Y-%m-%d',
                                  averagingPeriod=12, minPoints=10, tides=True,
                                  filters={},
                                  **kwargs):
        '''
        Compute velocity time series from JPL data

        Parameters
        ----------
        date1 : str or datetime
            First date in time series.
        date2 : str or datetime
            Last date in time series.
        dT : number
            Delta time for computing speed. (in hours)
        sampleInterval : number
            Frequency at which to compute estimates (hours).
        method : str, optional
            Use either point or regression. The default is 'regression'
        dateFormat : str, optional
            If date1/2 is a str, to datetime format. The default is '%Y-%m-%d'.
        minPoints : int, optional
            Return nan's if the number of valid points is < minPoits. The
            default is 10.
        averagingPeriod : int or float, optional
            The number of hours on either side of date1 & date2 to average
            when computing the two point positions. The default is 12 to
            yield the difference of two daily averages.
        Returns
        -------
        vx, vy, x, y: nparray
            velocity time series (vx, vy) with corresponding positions(x, y)
            with samples every sampleInterval hours.

        '''
        if method not in ['point', 'regression']:
            self.printError(f'Invalid method {method} keyword, must be point'
                            ' or regression')
        #
        # Convert to datetime if needed
        date1 = self._formatDate(date1, dateFormat=dateFormat)
        date2 = self._formatDate(date2, dateFormat=dateFormat)
        # Initialize
        currentDate = date1
        lastDate = currentDate + timedelta(hours=dT)
        vxSeries, vySeries, dateSeries, xSeries, ySeries = [], [], [], [], []
        #
        # Loop to compute velocities at sample interval.
        while lastDate < date2:
            vx, vy, x, y = self.computeVelocity(currentDate,
                                                lastDate,
                                                method=method,
                                                minPoints=minPoints,                
                                                filters=filters,
                                                averagingPeriod=averagingPeriod,
                                                tides=tides
                                                )
            #
            dateSeries.append(currentDate + timedelta(hours=dT/2))
            vxSeries.append(vx)
            vySeries.append(vy)
            xSeries.append(x)
            ySeries.append(y)
            currentDate = currentDate + timedelta(hours=sampleInterval)
            lastDate = currentDate + timedelta(hours=dT)
        # Done, return date, vx, vy, x, y
        return np.array(dateSeries), np.array(vxSeries), np.array(vySeries), \
            np.array(xSeries), np.array(ySeries)

    @catchError
    def _formatDate(self, date, dateFormat='%Y-%m-%d', **kwargs):
        '''
        Format dates as str to datetime

        Parameters
        ----------
        date : str or datetime
            date
        dateFormat : str, optional
            If date is a str, the corresponding format use to convert to
            datetime. The default is '%Y-%m-%d'.

        Returns
        -------
        date as datetime.

        '''
        if type(date) is str:
            return datetime.strptime(date, dateFormat)
        return date

    @catchError
    def _datetimeToDecimalYear(self, date, **kwargs):
        '''
        Convert date time to decimal year

        Parameters
        ----------
        date : datetime
            date to be converted to decimal year
        Returns
        -------
        date as datetime.
        '''
        yearLength = self.yearLengthLookupSeconds[date.year]
        return date.year + \
            (date - datetime(date.year, 1, 1)).total_seconds() / yearLength

    @catchError
    def _DecimalYearToDatetime(self, date, **kwargs):
        '''
        Convert date time to decimal year

        Parameters
        ----------
        date : datetime
            Date decimal year format (e.g., 2024.21)

        Returns
        -------
        date as datetime.
        '''
        year = int(date)
        fracYear = date - year
        yearLength = self.yearLengthLookupSeconds[year]
        return datetime(year, 1, 1) + timedelta(seconds=fracYear * yearLength)

    @catchError
    def _DecimalYearToDOYVector(self, date, **kwargs):
        '''
        Convert decimal year to doy for an array of decimal years

        Parameters
        ----------
        date : np.array of datetimes
            dates to be converted
        Returns
        -------
        dates as day of year (DOY).

        '''
        # Compute year, frac year, and length of year
        year = date.astype(int)
        fracYear = date - year
        yearLength = np.array([self.yearLengthLookupDays[y] for y in year])
        # Compute doy
        return (fracYear * yearLength).astype(int) + 1

    @catchError
    def subsetXYZ(self, date1, date2, dateFormat='%Y-%m-%d %H:%M:%S',
                  minPoints=1, removeOverlap=True, sigmaMultiple=True,
                  quiet=True, filters={}, **kwargs):
        '''
        Return all x,y, z points in interval [date1, date2]

        Parameters
        ----------
        date1 : str or datetime
            First date datetime or ascii with formate specified by dateFormat.
        date2 : str or datetime
            Last date datetime or ascii with formate specified by dateFormat.
        dateFormat : datetime format str, optional
            Format for conversion to date time. The default is
            '%Y-%m-%d %H:%M:%S'.
        minPoints : int, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        removeOverlap : bool, optional
            Return just data for the 24 hour period corresponding to each day
            file (i.e., removes the overlap between files).
            The default is True.
        sigmaMultiple : int, optional.
            Discard outliers > sigmaMuliple * sigma. Use None for no removal.
            The default is 3.
        quiet : bool
            Suppress warning messages. The default is True.
        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        # Convert to datetime if needed
        date1 = self._formatDate(date1, dateFormat=dateFormat, **kwargs)
        date2 = self._formatDate(date2, dateFormat=dateFormat, **kwargs)
        #
        if self.DB is None and self.DBData is None:
            return self._subsetXYZtext(date1, date2, minPoints=minPoints,
                                       quiet=quiet, **kwargs)
        return self._subsetXYZDB(date1, date2, minPoints=minPoints,
                                 quiet=quiet, filters=filters, **kwargs)

    @catchError
    def _subsetXYZDB(self, date1, date2, dateFormat='%Y-%m-%d %H:%M:%S',
                     minPoints=1, removeOverlap=True, sigmaMultiple=3,
                     quiet=True, filters={}, tides=True, **kwargs):
        '''
        Return all x, y, z points in interval [date1, date2]

        Parameters
        ----------
        date1 : str or datetime
            First date datetime or ascii with formate specified by dateFormat.
        date2 : str or datetime
            Last date datetime or ascii with formate specified by dateFormat.
        dateFormat : datetime format str, optional
            Format for conversion to date time. The default is
            '%Y-%m-%d %H:%M:%S'.
        minPoints : int, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        removeOverlap : bool, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        sigmaMultiple : int, optional.
            Discard outliers > sigmaMuliple * sigma. Use None for no removal.
            The defaul is 3.
        quiet : bool
            Suppress warning messages. The default is True.
        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        # Use decimal dates for DB queery
        d1 = self._datetimeToDecimalYear(date1, **kwargs)
        d2 = self._datetimeToDecimalYear(date2, **kwargs)

        # Query data base for station data
        if self.DBData is None:
            data = self.DB.getStationDateRangeData(self.stationName, d1, d2,
                                                   'landice', 'gps_data',
                                                   filters=filters)
        else:
            data = self.DBData[(self.DBData['decimal_year'] >= d1) & 
                               (self.DBData['decimal_year'] <= d2)]
    
        # This removes the overlap that comes with the GPS day files
        if removeOverlap:
            data = self._removeOverlap(data)
        # Check if there is data
        if len(data) == 0:
            if not quiet:
                print(f'no data in date range for {self.stationName}')
            return np.nan, np.nan, np.nan, np.nan, np.nan
        epoch = data['decimal_year'].to_numpy()
        # Make sure all is in place for coordinate conversion
    
        # height_arp_no_Tides has solid_earth_tide + load_tide_fes + pole_tide removed.
        # For floating ice shelves, ocean_tide_fes would also need subtracting (future work).
        # No inverse barometer column is present in the current schema.
        zCol = 'height_arp_no_tides' if tides else 'height_arp'
        lat, lon, z = [data[x].to_numpy() for x in ['lat', 'lon', zCol]]
        if self.epsg is None:
            self._determineEPSG(data['lat'].to_numpy()[0])
        #
        if self.lltoxy is None or self.eceftoll is None or self.lltoecef is None:
            self._initCoordinateConversion()
        # Convert to x, y
        x, y = self.lltoxy(lat, lon)
        #
        date = np.array([self._DecimalYearToDatetime(d) for d in data['decimal_year']])
        #
        if sigmaMultiple is not None:
            x, y, z, epoch, date = self.removeOutliers(x, y, z, epoch, date,
                                                       sigmaMultiple=sigmaMultiple,
                                                       **kwargs)
        #
        
        # Lat dependendent length scale for projected meters to real meters
        self.meanLat = np.mean(data['lat'].to_numpy())
        self.projLengthScale = self.proj.get_factors(0, self.meanLat
                                                     ).parallel_scale
        return date, x, y, z, epoch

    @catchError
    def removeOutliers(self, x, y, z, epoch, date, sigmaMultiple=3, **kwargs):
        '''
        Remove data where date is not equal to the nominal doy to remove
        overlap
        x, y, z : nparray
            Projected x, y, z positions.
        epoch, date : times
            epoch and ate
        returns:
            Filtered versions of x, y, z, epoch, date.
        '''
        good = np.ones(x.shape, dtype=bool)  # Initially keep all
        # Loop through variables and detect outlier as sigmaMultiple deviation
        # from a line fit.
        for d in [x, y, z]:
            fitResult = linregress(epoch, d)
            fit = fitResult[0] * epoch + fitResult[1]
            detrended = d - fit
            sigma = np.std(detrended)
            good = np.logical_and(good,
                                  np.abs(detrended) < sigmaMultiple*sigma)
        return x[good], y[good], z[good], epoch[good], date[good]

    @catchError
    def _removeOverlap(self, data, **kwargs):
        '''
        Remove data where date is not equal to the nominal doy to remove
        overlap. This is because 24 hour GPS dayfiles have several hours
        overlap before and after the nominal 24 hour period.

        Parameters
        ----------
        x, y : nparray
            Projected x and y positions.
        data : Pandas dataframe
            Dataframe with data returned from database.
        returns:
            Filtered versions of data.
        '''
        # new schema: continuous data stream, no day-file overlap to remove
        return data

    @catchError
    def _subsetXYZtext(self, date1, date2, minPoints=1, dateFormat='%Y-%m-%d',
                       quiet=True, **kwargs):
        '''
        Return all x, y, z points in interval [date1, date2] for text input.

        Parameters
        ----------
        date1 : str or datetime
            First date in date range.
        date2 : str or datetime
            Second date in date range.
        minPoints : TYPE, optional
            Return nan's if # of valid points is < minPoits. The default is 1.
        dateFormat : datetime format str, optional
            Format for conversion to date time. The default is '%Y-%m-%d'.
              quiet : bool
        Suppress warning messages. The default is True.
        Returns
        -------
        x, y, z np.array
            x, y, z values in projected coordinates.

        '''
        # Convert to datetime if needed
        date1 = self._formatDate(date1, dateFormat=dateFormat, **kwargs)
        date2 = self._formatDate(date2, dateFormat=dateFormat, **kwargs)
        inRange = np.logical_and(self.date >= date1, self.date <= date2)
        if inRange.sum() < minPoints:
            if not quiet:
                print(f'no data in date range for {self.stationName}')
            return np.nan, np.nan, np.nan
        #
        return self.date[inRange], self.x[inRange], self.y[inRange], \
            self.z[inRange], self.epoch[inRange]
