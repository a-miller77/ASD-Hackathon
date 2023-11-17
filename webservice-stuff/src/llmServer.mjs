import express from "express";

const app = new express();
const port = 3000;

app.use(express.text());

app.use(express.static("public", {index: "index.html"}))

app.listen(3000, () => {
console.log("listening on http://localhost:3000")
});