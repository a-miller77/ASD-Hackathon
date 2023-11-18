const server = 'localhost';
const getResponseURL = `http://${server}:3000/activeResponse`;
const getLastResponseURL = `http://${server}:3000/lastResponse`;
const getTermsURL = `http://${server}:3000/terms`

const getResponse = (socketId) => {
    return fetch(getResponseURL + "?id=" + socketId, {
        method: "GET"
    })
        .then((response) => {
        return response.json()
            .then((data) => {
            return data.curLength;
        });
    });
}

const getLastResponse = (socketId) => {
    return fetch(getLastResponseURL + "?id=" + socketId, {
        method: "GET"
    })
        .then((response) => {
            return response.json().then((json) => {
                return json.lastResponse;
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