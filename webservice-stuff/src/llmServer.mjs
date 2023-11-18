import express from "express";
import {createServer} from "node:http";
import {Server} from "socket.io";


const app = new express();
const server = createServer(app);
const io = new Server(server);

const port = 3000;

let conversations = {};

app.use(express.text());

app.use(express.static("public", {index: "index.html"}))

app.get('/activeConversations', (req, res) => {
    res.json( {
        conversations
    })
});

app.get('/activeResponse', (req, res) => {
    let socketId = req.query.id;
    // Call internally to get a response from our model
    let response = "Bot Example Response";
    res.json({response});
});

app.post('/input', (req, res) => {

});

app.delete('/resetConv', (req, res) => {

});

io.on('connection', (socket) => {
    console.log("User connected");
    conversations[socket.id] = [];

    socket.on('disconnect', () => {
        delete conversations[socket.id];
        console.log("User disconnected");
    });

    socket.on('userMessage', (data) => {
        conversations[socket.id].push(['User', data.message]);
        console.log(data);
        console.log(socket.id);
    });
});

server.listen(port, () => {
    console.log("listening on http://localhost:3000");
});