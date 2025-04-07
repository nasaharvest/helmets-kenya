// Distribution of all checked points
// Author: izvonkov@umd.edu

var df = ee.FeatureCollection('users/izvonkov/Helmets_Kenya_v1');
var crops = df.filter(ee.Filter.eq('is_crop', 1));
var maize = crops.filter(ee.Filter.eq('crop_type', 'maize'));
var sugarcane = crops.filter(ee.Filter.eq('crop_type', 'sugarcane'));
var banana = crops.filter(ee.Filter.eq('crop_type', 'banana'));
var wheat = crops.filter(ee.Filter.eq('crop_type', 'wheat'));
var remaining = ee.List([
    'Tea',
    'tea',
    'beans',
    'sunflower',
    'rice',
    'soybean',
    'cassava',
]);

var other = crops.filter(ee.Filter.inList('crop_type', remaining));
print('All', df.size());
print('Total', crops.size()); // 4925
print('Maize', maize.size()); // 4351
print('Sugarcane', sugarcane.size()); // 301
print('Banana', banana.size()); // 140
print('Wheat', wheat.size()); // 106
print('Other', other.size()); // 25
Map.addLayer(maize, { color: 'orange' }, '🟠 Maize');
Map.addLayer(sugarcane, { color: 'green' }, '🟢 Sugarcane');
Map.addLayer(banana, { color: 'yellow' }, '🟡 Banana');
Map.addLayer(wheat, { color: 'brown' }, '🟤 Wheat');
Map.addLayer(other, { color: 'red' }, '🔴 Remaining Crops');

Map.centerObject(df, 8);
