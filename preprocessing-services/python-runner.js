let {PythonShell} = require('python-shell')
var fs = require('fs');
var path = require('path');
var axios = require('axios');

var scriptRun = function(pythonJob, worker){
    console.log(worker);
    worker.status = 'busy';
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
                            console.log(`Worker let let the Boss know job is complete.`);
                            worker.status = 'bored';
                        }).catch((error) => {
                            worker.status = 'bored'
                        });
                    } else {
                        worker.status = 'bored'
                    }
                } catch (err) {
                   worker.status = 'bored'
                }
            });
            resolve({'message': 'job started'});
        } catch (err) {
            resolve(err)
            worker.status = 'bored'
        }
    });
}
module.exports = {scriptRun}