function sendMessage() {
    let userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    let chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p class="user-message"><strong>You:</strong> ${userInput}</p>`;

    fetch("/chat", {
        method: "POST",
        body: JSON.stringify({ message: userInput }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<p class="bot-message"><strong>Bot:</strong> ${data.response}</p>`;
        document.getElementById("user-input").value = "";
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}
