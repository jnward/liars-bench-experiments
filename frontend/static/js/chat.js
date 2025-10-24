// Chat state
let messages = [];
let chatTitle = '';
let systemPrompt = '';
let isLoading = false;
let editingIndex = -1;

// DOM Elements
const chatMessagesDiv = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const systemPromptInput = document.getElementById('system-prompt');
const chatTitleInput = document.getElementById('chat-title');
const saveSnapshotBtn = document.getElementById('save-snapshot-btn');
const snapshotSelector = document.getElementById('snapshot-selector');
const loadSnapshotBtn = document.getElementById('load-snapshot-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const messageArea = document.getElementById('message-area');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadCurrentChat();
    await loadSnapshotsList();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    saveSnapshotBtn.addEventListener('click', saveSnapshot);
    loadSnapshotBtn.addEventListener('click', loadSnapshot);
    newChatBtn.addEventListener('click', startNewChat);

    // Auto-update title based on system prompt if title is empty
    systemPromptInput.addEventListener('input', () => {
        if (!chatTitleInput.value.trim()) {
            const firstChars = systemPromptInput.value.slice(0, 50).trim();
            if (firstChars) {
                chatTitleInput.value = firstChars;
            }
        }
    });

    // Update title when changed
    chatTitleInput.addEventListener('input', () => {
        chatTitle = chatTitleInput.value;
    });

    // Update system prompt when changed
    systemPromptInput.addEventListener('input', () => {
        systemPrompt = systemPromptInput.value;
    });
}

// Load current chat state
async function loadCurrentChat() {
    try {
        const response = await fetch('/api/chat/current');
        const data = await response.json();

        if (data.messages && data.messages.length > 0) {
            messages = data.messages;
            chatTitle = data.title || '';
            chatTitleInput.value = chatTitle;

            // Extract system prompt from first message if it exists
            if (messages.length > 0 && messages[0].role === 'system') {
                systemPrompt = messages[0].content;
                systemPromptInput.value = systemPrompt;
            }

            renderMessages();
        }
    } catch (error) {
        console.error('Error loading current chat:', error);
    }
}

// Save current state
async function saveCurrentState() {
    try {
        const chatData = {
            title: chatTitle || chatTitleInput.value || 'Untitled',
            timestamp: new Date().toISOString(),
            messages: buildMessagesArray()
        };

        await fetch('/api/chat/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(chatData)
        });
    } catch (error) {
        console.error('Error saving state:', error);
    }
}

// Build messages array with current system prompt
function buildMessagesArray() {
    const msgs = [];
    const currentSystemPrompt = systemPromptInput.value.trim();

    // Always include system prompt as first message
    if (currentSystemPrompt) {
        msgs.push({ role: 'system', content: currentSystemPrompt });
    }

    // Add all user and assistant messages (skip old system messages)
    messages.forEach(msg => {
        if (msg.role !== 'system') {
            msgs.push(msg);
        }
    });

    return msgs;
}

// Send message
async function sendMessage() {
    const userMessage = userInput.value.trim();

    if (!userMessage) {
        showMessage('Please enter a message', 'warning');
        return;
    }

    if (isLoading) {
        showMessage('Please wait for the current response', 'warning');
        return;
    }

    // Add user message to messages
    messages.push({ role: 'user', content: userMessage });
    userInput.value = '';

    // Update title if empty
    if (!chatTitle && !chatTitleInput.value) {
        const title = systemPromptInput.value.slice(0, 50).trim() || 'Untitled';
        chatTitle = title;
        chatTitleInput.value = title;
    }

    renderMessages();

    // Show loading indicator
    isLoading = true;
    sendBtn.disabled = true;
    sendBtn.textContent = 'Sending...';

    try {
        // Build messages array with system prompt
        const apiMessages = buildMessagesArray();

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: apiMessages,
                title: chatTitle || chatTitleInput.value
            })
        });

        const data = await response.json();

        if (data.error) {
            showMessage('Error: ' + data.error, 'error');
            // Remove the user message if the request failed
            messages.pop();
        } else {
            // Add assistant response
            messages.push({ role: 'assistant', content: data.message });
            showMessage('Response received', 'success');
        }

        renderMessages();

    } catch (error) {
        showMessage('Error sending message: ' + error.message, 'error');
        // Remove the user message if the request failed
        messages.pop();
        renderMessages();
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
    }
}

// Render all messages
function renderMessages() {
    chatMessagesDiv.innerHTML = '';

    // Only render user and assistant messages (system prompt is separate)
    const displayMessages = messages.filter(msg => msg.role !== 'system');

    displayMessages.forEach((msg, index) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${msg.role}-message`;

        const roleLabel = document.createElement('div');
        roleLabel.className = 'message-role';
        roleLabel.textContent = msg.role === 'user' ? 'You' : 'Assistant';

        // Check if this message is being edited
        const isEditing = editingIndex === index;

        if (isEditing) {
            // Show textarea for editing
            const editArea = document.createElement('textarea');
            editArea.className = 'message-edit-area';
            editArea.value = msg.content;
            editArea.rows = 5;

            const editActionsDiv = document.createElement('div');
            editActionsDiv.className = 'message-actions';

            // If it's a user message, show "Save & Resend" button
            if (msg.role === 'user') {
                const resendBtn = document.createElement('button');
                resendBtn.className = 'btn-action btn-resend';
                resendBtn.textContent = 'Save & Resend';
                resendBtn.onclick = () => saveAndResend(index, editArea.value);
                editActionsDiv.appendChild(resendBtn);
            }

            const saveBtn = document.createElement('button');
            saveBtn.className = 'btn-action btn-save';
            saveBtn.textContent = 'Save';
            saveBtn.onclick = () => saveEdit(index, editArea.value);

            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn-action';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.onclick = () => cancelEdit();

            editActionsDiv.appendChild(saveBtn);
            editActionsDiv.appendChild(cancelBtn);

            messageDiv.appendChild(roleLabel);
            messageDiv.appendChild(editArea);
            messageDiv.appendChild(editActionsDiv);
        } else {
            // Show normal message content
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = msg.content;

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'message-actions';

            // Edit button
            const editBtn = document.createElement('button');
            editBtn.className = 'btn-action';
            editBtn.textContent = 'Edit';
            editBtn.onclick = () => startEdit(index);

            // Delete from here button
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn-action btn-danger';
            deleteBtn.textContent = 'Delete from here';
            deleteBtn.onclick = () => deleteFromHere(index);

            actionsDiv.appendChild(editBtn);
            actionsDiv.appendChild(deleteBtn);

            messageDiv.appendChild(roleLabel);
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(actionsDiv);
        }

        chatMessagesDiv.appendChild(messageDiv);
    });

    // Scroll to bottom
    chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
}

// Start editing a message
function startEdit(index) {
    editingIndex = index;
    renderMessages();
}

// Save edited message
function saveEdit(index, newContent) {
    if (newContent.trim() === '') {
        showMessage('Message cannot be empty', 'warning');
        return;
    }

    // Filter out system messages for display index
    const displayMessages = messages.filter(msg => msg.role !== 'system');
    const actualIndex = messages.findIndex(msg => msg === displayMessages[index]);

    messages[actualIndex].content = newContent.trim();
    editingIndex = -1;
    renderMessages();
    saveCurrentState();
}

// Cancel editing
function cancelEdit() {
    editingIndex = -1;
    renderMessages();
}

// Save edit and resend from this point
async function saveAndResend(index, newContent) {
    if (newContent.trim() === '') {
        showMessage('Message cannot be empty', 'warning');
        return;
    }

    if (isLoading) {
        showMessage('Please wait for the current response', 'warning');
        return;
    }

    // Filter out system messages for display index
    const displayMessages = messages.filter(msg => msg.role !== 'system');
    const actualIndex = messages.findIndex(msg => msg === displayMessages[index]);

    // Update the message content
    messages[actualIndex].content = newContent.trim();

    // Delete all messages after this point
    messages = messages.slice(0, actualIndex + 1);

    // Clear editing state
    editingIndex = -1;

    // Update title if empty
    if (!chatTitle && !chatTitleInput.value) {
        const title = systemPromptInput.value.slice(0, 50).trim() || 'Untitled';
        chatTitle = title;
        chatTitleInput.value = title;
    }

    renderMessages();

    // Show loading indicator
    isLoading = true;
    sendBtn.disabled = true;
    sendBtn.textContent = 'Sending...';

    try {
        // Build messages array with system prompt
        const apiMessages = buildMessagesArray();

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: apiMessages,
                title: chatTitle || chatTitleInput.value
            })
        });

        const data = await response.json();

        if (data.error) {
            showMessage('Error: ' + data.error, 'error');
        } else {
            // Add assistant response
            messages.push({ role: 'assistant', content: data.message });
            showMessage('Response received', 'success');
        }

        renderMessages();

    } catch (error) {
        showMessage('Error sending message: ' + error.message, 'error');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
    }
}

// Delete message and all subsequent messages
function deleteFromHere(index) {
    // Filter out system messages for display index
    const displayMessages = messages.filter(msg => msg.role !== 'system');
    const actualIndex = messages.findIndex(msg => msg === displayMessages[index]);

    if (confirm('Delete this message and all messages after it?')) {
        messages = messages.slice(0, actualIndex);
        renderMessages();
        saveCurrentState();
    }
}

// Save snapshot
async function saveSnapshot() {
    try {
        const title = chatTitleInput.value || 'Untitled';
        const chatData = {
            title: title,
            timestamp: new Date().toISOString(),
            messages: buildMessagesArray()
        };

        const response = await fetch('/api/snapshot/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(chatData)
        });

        const data = await response.json();

        if (data.error) {
            showMessage('Error saving snapshot: ' + data.error, 'error');
        } else {
            showMessage('Snapshot saved: ' + data.filename, 'success');
            await loadSnapshotsList();
        }
    } catch (error) {
        showMessage('Error saving snapshot: ' + error.message, 'error');
    }
}

// Load snapshots list
async function loadSnapshotsList() {
    try {
        const response = await fetch('/api/snapshots');
        const data = await response.json();

        snapshotSelector.innerHTML = '<option value="">Load snapshot...</option>';

        data.snapshots.forEach(snapshot => {
            const option = document.createElement('option');
            option.value = snapshot.filename;
            // Format the display name
            const displayName = snapshot.filename
                .replace('chat_snapshot_', '')
                .replace('.json', '')
                .replace(/_/g, ' ');
            option.textContent = displayName;
            snapshotSelector.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading snapshots:', error);
    }
}

// Load snapshot
async function loadSnapshot() {
    const filename = snapshotSelector.value;

    if (!filename) {
        showMessage('Please select a snapshot', 'warning');
        return;
    }

    try {
        const response = await fetch(`/api/snapshot/load/${filename}`);
        const data = await response.json();

        if (data.error) {
            showMessage('Error loading snapshot: ' + data.error, 'error');
            return;
        }

        // Load the snapshot data
        messages = data.messages || [];
        chatTitle = data.title || '';
        chatTitleInput.value = chatTitle;

        // Extract system prompt
        if (messages.length > 0 && messages[0].role === 'system') {
            systemPrompt = messages[0].content;
            systemPromptInput.value = systemPrompt;
        } else {
            systemPromptInput.value = '';
        }

        renderMessages();
        showMessage('Snapshot loaded', 'success');

    } catch (error) {
        showMessage('Error loading snapshot: ' + error.message, 'error');
    }
}

// Start new chat
function startNewChat() {
    if (messages.length > 0) {
        if (!confirm('Start a new chat? Current chat will be saved.')) {
            return;
        }
    }

    messages = [];
    chatTitle = '';
    chatTitleInput.value = '';
    systemPromptInput.value = '';
    chatMessagesDiv.innerHTML = '';

    showMessage('New chat started', 'success');
}

// Show message to user
function showMessage(message, type = 'info') {
    messageArea.textContent = message;
    messageArea.className = `message ${type}`;
    messageArea.style.display = 'block';

    setTimeout(() => {
        messageArea.style.display = 'none';
    }, 3000);
}
