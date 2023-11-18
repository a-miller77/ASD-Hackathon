const server = 'localhost';
const getResponseURL = `http://${server}:3000/activeResponse`;

const getResponse = async(socketId) => {
    return fetch(getResponseURL + "?id=" + socketId, {
        method: "GET"
    }).then((response) => {
        return response.json().then((data) => {
            return data.response;
        });
    });
}