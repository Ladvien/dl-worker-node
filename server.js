var express = require('express');
var bodyParser = require('body-parser');
var pythonRunner = require('./preprocessing-services/python-runner');

var app = express();
const port = 3000;

app.use(bodyParser.json())

// Python script runner interface
app.post('/scripts/run', (req, res) => {
    try {
        let pythonJob = req.body;
        pythonRunner.scriptRun(pythonJob)
        .then((response, rejection) => {
            res.send(response);
        });
    } catch (err) {
        res.send(err);
    }
});

app.listen(port, () => {
    console.log(`Started on port ${port}`);
});