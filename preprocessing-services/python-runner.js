let {PythonShell} = require('python-shell')
var fs = require('fs');
var path = require('path');

var scriptRun = function(pythonJob){    
    return new Promise((resolve, reject) => {
        try {
            let options = {
                mode: 'text',
                pythonOptions: ['-u'], // get print results in real-time
                scriptPath: path.relative(process.cwd(), 'python-scripts/'),
                args: [JSON.stringify(pythonJob)]
            };
            PythonShell.run(pythonJob.scriptName, options, function (err, results) {
                if (err) throw err;
                try {
                    result = JSON.parse(results.pop());
                    if(result) {
                        resolve(result);
                    } else {
                        reject({'err': ''})
                    }
                } catch (err) {
                    reject({'error': 'Failed to parse Python script return object.'})
                }
            });
        } catch (err) {
            reject(err)
        }
    });
}
module.exports = {scriptRun}