var fs = require('fs');


var readJsonFromFile = function(dataPath) {
    let rawdata = fs.readFileSync(dataPath);  
    return JSON.parse(rawdata);
}

// var readJsonFromFileAsync = function(dataPath) {
//     return new Promise((resolve, reject) => {
//         try {
//             let rawdata = fs.readFileSync(dataPath);  
//             let data = JSON.parse(rawdata);  
//             resolve(data);
//         } catch (err) {
//             reject(err);
//         }
//     });
// }


module.exports = {readJsonFromFile}