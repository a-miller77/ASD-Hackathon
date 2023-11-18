import express from "express";
import {createServer} from "node:http";
import {Server} from "socket.io";
import { parse } from "csv-parse";
import fs from "node:fs";



// Parse CSV

let terms = [];
fs.createReadStream("./Hacksgiving ASD V1.4-Terms.csv")
    .pipe(parse({delimiter: ",", from_line: 2}))
    .on("data", (row) => {
        terms.push(row);
    })
    .on("error", (err) => {
        console.log(err.message);
    });

// Server

const app = new express();
const server = createServer(app);
const io = new Server(server);

const port = 3000;

let conversations = {};

app.use(express.json());

app.use(express.static("public", {index: "index.html"}))

app.get('/activeConversations', (req, res) => {
    res.json( {
        conversations
    })
});

app.get('/activeResponse', async (req, res) => {
    let socketId = req.query.id;
    // console.log(conversations[socketId]);
    // let curLength = conversations[socketId].length;
    res.json({socketId});
});

app.get('/lastResponse', async (req, res) => {
    let socketId = req.query.id;
    if (conversations[socketId]) {
        let lastResponse = conversations[socketId].slice(-1)[0];
        console.log("User: " + socketId + " is waiting for a response. . .");
        res.json({lastResponse});
    } else {
        let status = 'error';
        res.json({status});
    }
})

app.get('/terms', (req, res) => {
    res.json({terms});
})

app.post('/input', (req, res) => {
    let socketId = req.query.id;
    if (conversations[socketId]) {
        conversations[socketId].push(req.body.message);
        let status = "Success";
        res.json({
            status
        })
    } else {
        let status = "error";
        res.json({
            status
        })
    }
});

app.delete('/resetConv', (req, res) => {
    let socketId = req.query.id;
    conversations[socket.id] = [];

    let status = "Success";
    res.json({
        status
    });
});

io.on('connection', (socket) => {
    console.log("User connected");
    conversations[socket.id] = [];

    socket.on('disconnect', () => {
        delete conversations[socket.id];
        console.log("User disconnected");
    });

    socket.on('userMessage', (data) => {
        conversations[socket.id].push(data.message);
    });
});

server.listen(port, () => {
    console.log("listening on http://localhost:3000");
});