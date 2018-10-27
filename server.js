var express = require('express');
var bodyParser = require('body-parser');
var pythonRunner = require('./preprocessing-services/python-runner');
var schedule = require('node-schedule');
var axios = require('axios');
var fs = require('fs');
var {Worker} = require('./worker/worker');

// Get Worker Node configuration
var fs = require('fs');
var config = JSON.parse(fs.readFileSync('./python-scripts/worker-node-configure.json', 'utf8'));

if(!config) { 
    console.log('No configuration file found.')
    process.exit();
}

// Boss' address
bossAddress = config.bossAddress;
nodeName = config.nodeName;
console.log(`Boss's address is ${bossAddress}`);
console.log(`This worker's name is ${nodeName}`);

var worker = new Worker('bored');

// Start server and add Middleware
var app = express();
const port = 3000;
app.use(bodyParser.json())

// Start checking for Boredom
var j = schedule.scheduleJob('*/1 * * * *', function(){
    if (worker.status === 'bored') {
        console.log('Worker is bored.');
        axios({
            method: 'post',
            url: bossAddress + `/bored/${nodeName}`
        }).then((response) => {
            let orderId = response.data._id
            let jobId = response. data.jobId;
            console.log(`Boss provided jobID #${jobId}`);
            axios({
                method: 'get',
                url: bossAddress + `/retrieve/job/${jobId}`
            }).then((response) => {
                let job = response.data;
                console.log(`Worker found the details for jobID #${jobId}`);
                job.callbackAddress = bossAddress;
                job.assignmentId = orderId;
                pythonRunner.scriptRun(job, worker)
                .then((response) => {
                    console.log('Worker started job, will let Boss know when finished.');
                });
            }).catch((error) => {
                console.log(error);
            });
        }).catch((error) => {
            console.log('Failed to find new job.')
        });
    }
});

// Python script runner interface
app.post('/scripts/run', (req, res) => {
    try {
        let pythonJob = req.body;
        pythonRunner.scriptRun(pythonJob)
        .then((response) => {
            console.log(response);
            res.send(response);
        });
    } catch (err) {
        res.send(err);
    }
});

app.listen(port, () => {
    console.log(`Started on port ${port}`);
});