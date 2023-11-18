let userInputText;
let userInputBtn;
let socket = io();

window.onload = async () => {
    userInputText = document.getElementById("userInput");
    userInputBtn = document.getElementById("planeSendBtn");
    userInputText.addEventListener('keydown', (e) => submitQuery(e));
    userInputBtn.addEventListener('click', (e) => submitQuery(e))
}

const submitQuery = (event) => {
    if (event instanceof KeyboardEvent) {
        if (event.key === "Enter") {
            userInputText.disabled = true;
            userInputBtn.disabled = true;
            event.preventDefault();
            sendResponse().then(r => {
                userInputText.disabled = false;
                userInputBtn.disabled = false;
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
    let data = await getResponse(socket.id);

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