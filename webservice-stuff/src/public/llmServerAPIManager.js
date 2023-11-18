const server = 'localhost';
const getResponseURL = `http://${server}:3000/activeResponse`;
const getTermsURL = `http://${server}:3000/terms`

const getResponse = async(socketId) => {
    return fetch(getResponseURL + "?id=" + socketId, {
        method: "GET"
    }).then((response) => {
        return response.json().then((data) => {
            return data.response;
        });
    });
}

const getTerms = async () => {
    return fetch(getTermsURL, {
        method: "GET"
    }).then((response) => {
        return response.json().then((data) => {
            return data.terms;
        })
    })
}