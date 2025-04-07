/////////////////////////////////////////////////////////////////////////////////////
//
// Create maize map using S2 cloud free
//
// Authors: Ivan Zvonkov, Diana Frimpong
// Last Updated: Apr 3, 2025
//
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
// 1. Load administrative zone
/////////////////////////////////////////////////////////////////////////////////////
var gadm = ee.FeatureCollection('users/izvonkov/Kenya/gadm41_KEN_1');
var roi = gadm.filter(ee.Filter.eq('NAME_1', 'Nakuru'));
Map.centerObject(roi, 11);

/////////////////////////////////////////////////////////////////////////////////////
// 2a. Load Helmets points
/////////////////////////////////////////////////////////////////////////////////////
var df = ee.FeatureCollection('users/izvonkov/Helmets_Kenya_v1');
var points = df.filter(ee.Filter.eq('year', 2021));
var nakuruPoints = points.filterBounds(roi);

var corrective = ee.FeatureCollection(
    'users/izvonkov/Kenya/Corrective_2021_v1'
);
nakuruPoints = nakuruPoints.merge(corrective);
print('Corrective Points', corrective.size());

/////////////////////////////////////////////////////////////////////////////////////
// 2b. Load non-maize polygons manually drawn
/////////////////////////////////////////////////////////////////////////////////////
var nakuruNonMaizePolygons2021 = ee.FeatureCollection(
    'users/izvonkov/Kenya/Nakuru_non_maize_polygons_2021'
);

// Apply a negative buffer of 5m to allow removal of boundary effets
var bufferArgs = {
    proj: 'EPSG:32636',
    distance: -10.0,
    maxError: ee.ErrorMargin(0.0, 'projected'),
};
var nakuruNonMaizePolygons2021Buffered = nakuruNonMaizePolygons2021.map(
    function (feat) {
        var bufferedGeometry = feat.geometry().buffer(bufferArgs);
        return feat.setGeometry(bufferedGeometry);
    }
);

// Convert to FeatureCollection of S2 pixels
function getUTMZoneEPSG(centroid) {
    var point = centroid.coordinates();
    var lon = ee.Number(point.get(0));
    var lat = ee.Number(point.get(1));
    var utmZone = lon.add(180).divide(6).floor().add(1);
    var epsgCode = ee.Number(32600).add(utmZone);

    // Adjust for southern hemisphere
    epsgCode = ee.Algorithms.If(
        lat.lt(0),
        ee.Number(32700).add(utmZone),
        epsgCode
    );
    var epsgStr = ee.String('EPSG:').cat(ee.Number(epsgCode).format('%d'));
    return epsgStr;
}

var nakuruNonMaizePolygons2021BufferedWithEPSG =
    nakuruNonMaizePolygons2021Buffered.map(function (feat) {
        return feat.set(
            'projection',
            getUTMZoneEPSG(feat.geometry().centroid(0.1))
        );
    });

var collectionOfCollections = nakuruNonMaizePolygons2021BufferedWithEPSG.map(
    function (feat) {
        var proj = feat.get('projection');
        var reprojected = ee.Algorithms.ProjectionTransform(feat, proj, 0.1);
        var coveringGrid = reprojected.geometry().coveringGrid(proj, 10);

        var coveringPoints = coveringGrid.map(function (childFeat) {
            var centroid = childFeat.geometry().centroid(0.1, 'EPSG:4326');
            return childFeat.setGeometry(centroid);
        });

        var featProperties = feat.toDictionary();
        var multiplePoints = coveringPoints.map(function (childFeat) {
            return childFeat.set(featProperties);
        });
        return multiplePoints;
    }
);

var nonMaizeCropPoints2021 = ee
    .FeatureCollection(collectionOfCollections)
    .flatten();
print('Non maize crop points', nonMaizeCropPoints2021.size());
nakuruPoints = nakuruPoints.merge(nonMaizeCropPoints2021);

/////////////////////////////////////////////////////////////////////////////////////
// 3. Sentinel-2 L2A with S2 Cloudless
/////////////////////////////////////////////////////////////////////////////////////
function maskCloudProbability(image) {
    // Load the corresponding CLOUD_PROBABILITY image
    var cloudProbability = ee
        .ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filter(ee.Filter.eq('system:index', image.get('system:index')))
        .first(); // Match by metadata

    // Keep pixels with cloud probability < 30%
    var cloudMask = cloudProbability.lt(30);

    // Mask the original image
    return image.updateMask(cloudMask);
}

// Function to process images for a date range
function getCloudFreeMosaic(startDate, endDate) {
    var S2 = ee
        .ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(startDate, endDate)
        .filterBounds(roi)
        .map(maskCloudProbability); // Apply cloud probability mask

    // Create a mosaic and compute NDVI
    var mosaic = S2.median()
        .clip(roi)
        .addBands(S2.median().normalizedDifference(['B8', 'B4']).rename('NDVI')) // Add NDVI
        .select(['B2', 'B3', 'B4', 'B8', 'B8A', 'B9', 'B11', 'NDVI']); // Select bands

    return mosaic;
}

var dateRanges = [
    ['2021-04-01', '2021-06-01'], // April, May
    ['2021-06-01', '2021-08-01'], // June, July
    ['2021-08-01', '2021-10-01'], // August, September
    ['2021-10-01', '2021-12-01'], // October, November
];

// Map over date ranges to generate mosaics
var S2List = dateRanges.map(function (range) {
    return getCloudFreeMosaic(range[0], range[1]);
});

/////////////////////////////////////////////////////////////////////////////////////
// 4. Combine training labels with Earth observation image sequence
/////////////////////////////////////////////////////////////////////////////////////
var imageSequence = ee.ImageCollection.fromImages(S2List).toBands();
var bands = imageSequence.bandNames();
print(bands);
var training = imageSequence.sampleRegions({
    collection: nakuruPoints,
    properties: ['is_maize'],
    scale: 10,
});

print(
    'Maize training points',
    nakuruPoints.filter(ee.Filter.eq('is_maize', 1)).size()
);
print(
    'Non-maize training points',
    nakuruPoints.filter(ee.Filter.eq('is_maize', 0)).size()
);

/////////////////////////////////////////////////////////////////////////////////////
// 5. Train a classifier
/////////////////////////////////////////////////////////////////////////////////////
var trainedRf = ee.Classifier.smileRandomForest({ numberOfTrees: 50 })
    .setOutputMode('probability')
    .train({
        features: training,
        classProperty: 'is_maize',
        inputProperties: bands,
    });

print(trainedRf.explain());

var WC = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi);
var generousCropMask = WC.eq(40) // Crops
    .or(WC.eq(10)) // Tree cover
    .or(WC.eq(30)); // Grassland
Map.addLayer(generousCropMask, {}, 'Generous Crop Mask');

/////////////////////////////////////////////////////////////////////////////////////
// 7. Use classifier to make predictions
/////////////////////////////////////////////////////////////////////////////////////
var classifiedRf = imageSequence
    .select(bands)
    .mask(generousCropMask)
    .classify(trainedRf)
    .clip(roi);
var maizeMap = classifiedRf.unmask().clip(roi);

/////////////////////////////////////////////////////////////////////////////////////
// 8. Post-processing
/////////////////////////////////////////////////////////////////////////////////////
var maizeMapProcessed = maizeMap.focal_mode(30, 'square', 'meters');
var maizeMask = maizeMapProcessed.gt(0.5).rename('is_maize_map');
var classVis = { min: 0, max: 1.0, palette: ['yellow', 'green'] };
Map.addLayer(maizeMask, classVis, 'Maize Mask');

/////////////////////////////////////////////////////////////////////////////////////
// 9. Accuracy Assessment
/////////////////////////////////////////////////////////////////////////////////////
var fullDataset = ee.FeatureCollection(
    'users/izvonkov/Kenya/cop4geoglam_longrain_2021_processed'
);
var validationDataset = fullDataset.filterBounds(roi);
var validationHelmets = maizeMask.sampleRegions(validationDataset);
var errorMatrixHelmets = validationHelmets.errorMatrix(
    'is_maize',
    'is_maize_map'
);
print('Helmets Overall Accuracy: ' + errorMatrixHelmets.accuracy().getInfo());
print(
    "Helmets User's Accuracy: " +
        errorMatrixHelmets.consumersAccuracy().getInfo()[0][1]
);
print(
    "Helmets Producer's Accuracy: " +
        errorMatrixHelmets.producersAccuracy().getInfo()[1][0]
);
