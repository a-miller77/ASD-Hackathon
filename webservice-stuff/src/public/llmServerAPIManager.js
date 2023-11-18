const server = window.location.href;
const getResponseURL = `${server}activeResponse`;
const getLastResponseURL = `${server}lastResponse`;
const getTermsURL = `${server}terms`

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