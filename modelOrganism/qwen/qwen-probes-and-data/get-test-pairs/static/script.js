// Global state
let currentState = null;
let currentGeneration = null;
let thinkingExpanded = true;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadState();
});

// Load current state from server
async function loadState() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        currentState = data;

        if (data.complete) {
            showCompleteScreen();
        } else {
            updateUI();
            await generateResponse();
        }
    } catch (error) {
        console.error('Error loading state:', error);
        alert('Failed to load state. Please refresh the page.');
    }
}

// Update UI with current state
function updateUI() {
    // Update progress indicator
    const progressText = `Pair ${currentState.current_index + 1}/${currentState.total_pairs} - ${capitalizeFirst(currentState.current_type)} Prompt`;
    document.getElementById('progress-indicator').textContent = progressText;

    // Update scenario and prompt
    document.getElementById('scenario-title').textContent = currentState.scenario;
    document.getElementById('prompt-text').textContent = `System: ${currentState.system_prompt}\n\nUser: ${currentState.user_prompt}`;

    // Update sidebar
    updateSidebar();

    // Update previous button state
    const canGoPrevious = !(currentState.current_index === 0 && currentState.current_type === 'truth');
    document.getElementById('btn-previous').disabled = canGoPrevious ? false : true;
}

// Update sidebar with approval status
function updateSidebar() {
    const sidebarList = document.getElementById('sidebar-list');
    sidebarList.innerHTML = '';

    currentState.approved_list.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'sidebar-item';

        // Highlight current
        if (item.index === currentState.current_index) {
            div.classList.add('current');
        }

        // Make clickable
        div.onclick = () => navigateTo(item.index, 'truth');

        const title = document.createElement('div');
        title.className = 'sidebar-item-title';
        title.textContent = `${item.index + 1}. ${item.scenario}`;

        const status = document.createElement('div');
        status.className = 'sidebar-item-status';
        const truthCheck = item.truth_approved ? '<span class="check">✓</span>' : '○';
        const deceptionCheck = item.deception_approved ? '<span class="check">✓</span>' : '○';
        status.innerHTML = `Truth: ${truthCheck} | Deception: ${deceptionCheck}`;

        div.appendChild(title);
        div.appendChild(status);
        sidebarList.appendChild(div);
    });
}

// Generate response for current prompt
async function generateResponse() {
    showLoading();

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                system_prompt: currentState.system_prompt,
                user_prompt: currentState.user_prompt
            })
        });

        const data = await response.json();
        currentGeneration = data;

        showResponse();
    } catch (error) {
        console.error('Error generating response:', error);
        alert('Failed to generate response. Please try again.');
        hideLoading();
    }
}

// Show loading spinner
function showLoading() {
    document.getElementById('loading-spinner').style.display = 'flex';
    document.getElementById('response-container').style.display = 'none';
}

// Hide loading spinner
function hideLoading() {
    document.getElementById('loading-spinner').style.display = 'none';
}

// Show generated response
function showResponse() {
    hideLoading();

    document.getElementById('thinking-content').textContent = currentGeneration.thinking;
    document.getElementById('response-content').textContent = currentGeneration.response;

    document.getElementById('response-container').style.display = 'block';

    // Reset thinking expansion
    if (thinkingExpanded) {
        document.getElementById('thinking-content').classList.remove('collapsed');
        document.getElementById('thinking-toggle').classList.remove('collapsed');
    } else {
        document.getElementById('thinking-content').classList.add('collapsed');
        document.getElementById('thinking-toggle').classList.add('collapsed');
    }
}

// Toggle thinking section
function toggleThinking() {
    const content = document.getElementById('thinking-content');
    const toggle = document.getElementById('thinking-toggle');

    if (content.classList.contains('collapsed')) {
        content.classList.remove('collapsed');
        toggle.classList.remove('collapsed');
        thinkingExpanded = true;
    } else {
        content.classList.add('collapsed');
        toggle.classList.add('collapsed');
        thinkingExpanded = false;
    }
}

// Approve current response and continue
async function approve() {
    try {
        const response = await fetch('/api/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                thinking: currentGeneration.thinking,
                response: currentGeneration.response
            })
        });

        const data = await response.json();

        if (data.complete) {
            showCompleteScreen();
        } else {
            await loadState();
        }
    } catch (error) {
        console.error('Error approving response:', error);
        alert('Failed to approve response. Please try again.');
    }
}

// Regenerate current response
async function regenerate() {
    await generateResponse();
}

// Navigate to previous prompt
async function navigatePrevious() {
    let newIndex = currentState.current_index;
    let newType = currentState.current_type;

    if (newType === 'deception') {
        newType = 'truth';
    } else if (newIndex > 0) {
        newIndex -= 1;
        newType = 'deception';
    } else {
        // Already at first truth prompt
        return;
    }

    await navigateTo(newIndex, newType);
}

// Navigate to specific prompt
async function navigateTo(index, type) {
    try {
        const response = await fetch('/api/navigate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index, type })
        });

        await response.json();
        await loadState();
    } catch (error) {
        console.error('Error navigating:', error);
        alert('Failed to navigate. Please try again.');
    }
}

// Show completion screen
function showCompleteScreen() {
    document.getElementById('generation-screen').style.display = 'none';
    document.getElementById('complete-screen').style.display = 'block';
}

// Utility function
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
