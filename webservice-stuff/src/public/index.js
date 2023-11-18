let userInputText;
let userInputBtn;
let filterInputText;
let filterInputBtn;
let socket = io();
let terms;

const submitQuery = (event) => {
    if (event instanceof KeyboardEvent) {
        if (event.key === "Enter") {
            userInputText.disabled = true;
            // userInputBtn.disabled = true;
            event.preventDefault();
            sendResponse().then(r => {
                userInputText.disabled = false;
                // userInputBtn.disabled = false;
            });
        }
    } else {
        userInputText.disabled = true;
        userInputBtn.disabled = true;
        event.preventDefault();
        sendResponse().then(r => {
            userInputText.disabled = false;
            userInputBtn.disabled = false;
        });

    }
}

const sendResponse = async () => {

    //DOM Element creation with user input
    let userChatBubble = document.createElement("div");
    userChatBubble.setAttribute("class", "media media-chat media-chat-reverse");

    let userChatBubbleMediaBody = document.createElement("div");
    userChatBubbleMediaBody.setAttribute("class", "media-body");

    let userChatBubbleContent = document.createElement("p");
    userChatBubbleContent.innerText = userInputText.value;

    let bottomRail = document.getElementById("bottomRail");

    bottomRail.insertAdjacentElement("beforebegin", userChatBubble);
    userChatBubble.insertAdjacentElement("afterbegin", userChatBubbleMediaBody);
    userChatBubbleMediaBody.insertAdjacentElement("afterbegin", userChatBubbleContent);

    //API call to the LLM
    socket.emit("userMessage", {message: userInputText.value});
    // Checks to see if the length of the conversation has increased -
    // possible race condition but model takes a while to run.
    let currentResponse = await getLastResponse(socket.id);
    let passingResponse = await getLastResponse(socket.id);
    while (currentResponse === passingResponse) {
        await new Promise(resolve => setTimeout(resolve, 5000))
            .then(() => { console.log('Waiting for response from Server'); });
        passingResponse = await getLastResponse(socket.id)
    }
    console.log(passingResponse);
    data = passingResponse;

    // Generate DOM element based on api response
    let llmChatBubble = document.createElement("div");
    llmChatBubble.setAttribute("class", "media media-chat");

    let llmChatBubbleMediaBody = document.createElement("div");
    llmChatBubbleMediaBody.setAttribute("class", "media-body");

    let llmChatBubbleContent = document.createElement("p");
    llmChatBubbleContent.innerText = data;

    let llmAvatar = document.createElement("img");
    llmAvatar.setAttribute("class", "avatar");
    llmAvatar.setAttribute("src", "Windows_10_Default_Profile_Picture.svg");
    llmAvatar.setAttribute("width", "32");
    llmAvatar.setAttribute("height", "32");
    llmAvatar.setAttribute("alt", "avatar");

    bottomRail.insertAdjacentElement("beforebegin", llmChatBubble);
    llmChatBubble.insertAdjacentElement("beforeend", llmAvatar);
    llmChatBubble.insertAdjacentElement("beforeend", llmChatBubbleMediaBody);
    llmChatBubbleMediaBody.insertAdjacentElement("afterbegin", llmChatBubbleContent);

    // Clear user input
    userInputText.value = "";
}

const loadTerms = async () => {
    terms = await getTerms();
    printTableTerms(terms)
}

const printTableTerms = (items) => {
    // Initialize the HTML of the table
    let table =
        document.getElementsByTagName("tbody").namedItem("table1body");
    // Effectively "resets" the table be setting the inner text to be nothing.
    table.innerText = ""
    // Source adapted from:
    // https://www.tutorialspoint.com/How-to-add-rows-to-a-table-using-JavaScript-DOM
    let row = table.insertRow(-1);
    // Iterate through the result set, adding table rows and table data
    // Enhanced for loop for iterating through a given array with a proper format
    items.forEach((item) => {
            row.insertCell(0).innerText = item[0];
            row.insertCell(1).innerText = item[1];
            row.insertCell(2).innerText = item[2];
            row.insertCell(3).innerText = item[3];
            row = table.insertRow(-1);
        }
    );

}

const validItem = (item, filterCondition) => {
    let ret = false;
    //Switch statement for filter and appropriate conditions
    if (item.includes(filterCondition)) {
        ret = true;
    }
    return ret;
}

const updateTerms = (event) => {
    if (event.key === "Enter") {
        let filteredTerms = [];
        terms.forEach((item) => {
            if (validItem(item[0].toLowerCase(), filterInputText.value.toLowerCase())) {
                filteredTerms.push(item);
            }
        });
        printTableTerms(filteredTerms);
    }
}

const autofill = (event) => {
    const isButton = event.target.nodeName === "A";
    if (isButton) {
        userInputText.value = event.target.innerText;
    }
}

window.onload = async () => {
    userInputText = document.getElementById("userInput");
    // userInputBtn = document.getElementById("planeSendBtn");
    filterInputText = document.getElementById("filter");
    userInputText.addEventListener('keydown', (e) => submitQuery(e));
    // userInputBtn.addEventListener('click', (e) => submitQuery(e))
    filterInputText.addEventListener('keydown', (e) => updateTerms(e));

    document.getElementById("suggestedOptions").addEventListener('click', (e) => autofill(e));

    loadTerms();
}