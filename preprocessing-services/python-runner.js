let {PythonShell} = require('python-shell')
var fs = require('fs');
var path = require('path');
var axios = require('axios');

var scriptRun = function(pythonJob){
    return new Promise((resolve, reject) => {
        try {
            let callbackAddress = pythonJob.callbackAddress;
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
                        console.log(callbackAddress + '/callback')
                        axios({
                            method: 'post',
                            url: callbackAddress + '/callback',
                            data: result
                        }).then((response) => {
                            console.log(response.data);
                        }).catch((error) => {
                            console.log(error);
                        });
                    } else {
                        
                    }
                } catch (err) {
                   console.log(err)
                }
            });
            resolve({'message': 'job started'});
        } catch (err) {
            reject(err)
        }
    });
}
module.exports = {scriptRun}