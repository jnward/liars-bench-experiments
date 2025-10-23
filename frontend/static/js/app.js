// State management
let currentData = [];
let currentIndex = 0;
let currentMetadata = {};
let currentCSV = '';
let showHistory = false;
let showTokens = false;
let currentSearchQuery = '';

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    await loadCSVList();
    setupEventListeners();
});

// Load available CSV files
async function loadCSVList() {
    try {
        const response = await fetch('/api/csvs');
        const data = await response.json();
        const selector = document.getElementById('csv-selector');

        data.files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            selector.appendChild(option);
        });
    } catch (error) {
        showMessage('Error loading CSV list: ' + error.message, 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('load-btn').addEventListener('click', loadData);
    document.getElementById('prev-btn').addEventListener('click', () => navigateToIndex(currentIndex - 1));
    document.getElementById('next-btn').addEventListener('click', () => navigateToIndex(currentIndex + 1));
    document.getElementById('toggle-history-btn').addEventListener('click', toggleHistory);
    document.getElementById('toggle-tokens-btn').addEventListener('click', toggleTokens);
    document.getElementById('toggle-filters-btn').addEventListener('click', toggleFilters);
    document.getElementById('apply-filters-btn').addEventListener('click', applyFilters);
    document.getElementById('search-btn').addEventListener('click', performSearch);
    document.getElementById('clear-search-btn').addEventListener('click', clearSearch);

    // Allow Enter key for search
    document.getElementById('search-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
}

// Load data from selected CSV
async function loadData() {
    const selector = document.getElementById('csv-selector');
    const filename = selector.value;

    if (!filename) {
        showMessage('Please select a CSV file', 'error');
        return;
    }

    currentCSV = filename;
    await fetchData();
}

// Fetch data with current filters
async function fetchData() {
    try {
        const params = new URLSearchParams();

        const probeFilter = document.getElementById('probe-filter').value;
        const labelFilter = document.getElementById('label-filter').value;
        const categoryFilter = document.getElementById('category-filter').value;
        const sortBy = document.getElementById('sort-select').value;

        if (probeFilter) params.append('probe', probeFilter);
        if (labelFilter) params.append('label', labelFilter);
        if (categoryFilter) params.append('category', categoryFilter);
        if (sortBy) params.append('sort', sortBy);
        if (currentSearchQuery) params.append('search', currentSearchQuery);

        const response = await fetch(`/api/data/${currentCSV}?${params.toString()}`);
        const result = await response.json();

        if (result.error) {
            showMessage('Error: ' + result.error, 'error');
            return;
        }

        currentData = result.data;
        currentMetadata = result.metadata;
        currentIndex = 0;

        // Populate filter options
        populateFilters();

        // Display statistics
        displayStatistics();

        // Show controls and content
        document.getElementById('controls-panel').style.display = 'block';
        document.getElementById('main-content').style.display = 'block';

        // Display first datapoint
        if (currentData.length > 0) {
            displayDatapoint();
            showMessage(`Loaded ${currentData.length} datapoint(s)`, 'success');
        } else {
            showMessage('No datapoints found with current filters', 'warning');
        }

    } catch (error) {
        showMessage('Error loading data: ' + error.message, 'error');
    }
}

// Populate filter dropdowns
function populateFilters() {
    const probeFilter = document.getElementById('probe-filter');
    const labelFilter = document.getElementById('label-filter');
    const categoryFilter = document.getElementById('category-filter');

    // Clear existing options (except "All")
    probeFilter.innerHTML = '<option value="">All</option>';
    labelFilter.innerHTML = '<option value="">All</option>';
    categoryFilter.innerHTML = '<option value="">All</option>';

    // Add unique probes
    if (currentMetadata.probes) {
        currentMetadata.probes.forEach(probe => {
            const option = document.createElement('option');
            option.value = probe;
            option.textContent = probe;
            probeFilter.appendChild(option);
        });
    }

    // Add unique labels
    currentMetadata.labels.forEach(label => {
        const option = document.createElement('option');
        option.value = label;
        option.textContent = label;
        labelFilter.appendChild(option);
    });

    // Add unique categories
    currentMetadata.categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        categoryFilter.appendChild(option);
    });
}

// Display score statistics
function displayStatistics() {
    const statsPanel = document.getElementById('score-stats');
    const stats = currentMetadata.score_stats;

    if (!stats || (!stats.deceptive.mean && !stats.honest.mean)) {
        statsPanel.style.display = 'none';
        return;
    }

    // Show the panel
    statsPanel.style.display = 'block';

    // Update deceptive stats
    document.getElementById('deceptive-mean').textContent =
        stats.deceptive.mean != null ? stats.deceptive.mean.toFixed(3) : '-';
    document.getElementById('deceptive-min').textContent =
        stats.deceptive.min != null ? stats.deceptive.min.toFixed(3) : '-';
    document.getElementById('deceptive-max').textContent =
        stats.deceptive.max != null ? stats.deceptive.max.toFixed(3) : '-';

    // Update honest stats
    document.getElementById('honest-mean').textContent =
        stats.honest.mean != null ? stats.honest.mean.toFixed(3) : '-';
    document.getElementById('honest-min').textContent =
        stats.honest.min != null ? stats.honest.min.toFixed(3) : '-';
    document.getElementById('honest-max').textContent =
        stats.honest.max != null ? stats.honest.max.toFixed(3) : '-';

    // Update separation
    document.getElementById('separation').textContent =
        stats.separation != null ? stats.separation.toFixed(3) : '-';
}

// Navigate to specific index
function navigateToIndex(index) {
    if (index < 0 || index >= currentData.length) return;
    currentIndex = index;
    displayDatapoint();
}

// Display current datapoint
function displayDatapoint() {
    if (currentData.length === 0) return;

    const datapoint = currentData[currentIndex];

    // Update counter
    document.getElementById('datapoint-counter').textContent =
        `${currentIndex + 1} / ${currentData.length}`;

    // Update navigation buttons
    document.getElementById('prev-btn').disabled = currentIndex === 0;
    document.getElementById('next-btn').disabled = currentIndex === currentData.length - 1;

    // Update metadata
    document.getElementById('probe-name').textContent = datapoint.probe_name || 'N/A';
    document.getElementById('dataset-name').textContent = datapoint.dataset_name;
    document.getElementById('category').textContent = datapoint.category;
    document.getElementById('model-name').textContent = datapoint.model_name;
    document.getElementById('layer').textContent = datapoint.layer;
    document.getElementById('label').textContent = datapoint.label;

    // Update score with 0-centered color coding
    const scoreElement = document.getElementById('prompt-score');
    if (datapoint.prompt_level_score != null && !isNaN(datapoint.prompt_level_score)) {
        scoreElement.textContent = datapoint.prompt_level_score.toFixed(3);
        scoreElement.style.color = getScoreColor(datapoint.prompt_level_score);
    } else {
        scoreElement.textContent = 'N/A';
        scoreElement.style.color = '#95a5a6';
    }

    // Check if token scores are available and update button state
    const tokenBtn = document.getElementById('toggle-tokens-btn');
    const hasTokenScores = datapoint.token_level_scores && datapoint.token_level_scores.length > 0;

    if (!hasTokenScores) {
        tokenBtn.disabled = true;
        tokenBtn.title = 'Token scores not available for this datapoint';
        showTokens = false; // Reset token view if it was on
        tokenBtn.textContent = 'Show Token Scores';
    } else {
        tokenBtn.disabled = false;
        tokenBtn.title = '';
    }

    // Update assistant response
    displayAssistantResponse(datapoint);

    // Update conversation history if visible
    if (showHistory) {
        displayConversationHistory(datapoint);
    }
}

// Display assistant response
function displayAssistantResponse(datapoint) {
    const responseDiv = document.getElementById('assistant-response');

    if (showTokens) {
        // Display with token scores
        displayTokenizedResponse(datapoint);
    } else {
        // Display plain text with search highlighting
        responseDiv.innerHTML = highlightSearchText(datapoint.final_assistant_response);
    }
}

// Display tokenized response with color coding
function displayTokenizedResponse(datapoint) {
    const responseDiv = document.getElementById('assistant-response');
    responseDiv.innerHTML = '';

    // Check if token scores are available
    if (!datapoint.token_level_scores || datapoint.token_level_scores.length === 0) {
        responseDiv.innerHTML = highlightSearchText(datapoint.final_assistant_response);
        responseDiv.innerHTML += '<div style="margin-top: 10px; color: #856404; font-style: italic;">Token-level scores not available for this datapoint.</div>';
        return;
    }

    datapoint.token_level_scores.forEach(item => {
        const span = document.createElement('span');
        span.textContent = item.token;
        span.className = 'token';
        span.style.backgroundColor = getScoreColor(item.score, 0.3);
        span.title = `Score: ${item.score.toFixed(3)}`;
        responseDiv.appendChild(span);
    });
}

// Display conversation history
function displayConversationHistory(datapoint) {
    const historyDiv = document.getElementById('history-content');
    historyDiv.innerHTML = '';

    datapoint.conversation_history.forEach(turn => {
        const turnDiv = document.createElement('div');
        turnDiv.className = `turn ${turn.role}`;

        const roleLabel = document.createElement('strong');
        roleLabel.textContent = turn.role.charAt(0).toUpperCase() + turn.role.slice(1) + ': ';

        const content = document.createElement('span');
        content.innerHTML = highlightSearchText(turn.content);

        turnDiv.appendChild(roleLabel);
        turnDiv.appendChild(content);
        historyDiv.appendChild(turnDiv);
    });
}

// Get color based on score (red = high deception, green = low/honest)
function getScoreColor(score, alpha = 1) {
    // Handle null/undefined scores
    if (score == null || isNaN(score)) {
        return `rgba(149, 165, 166, ${alpha})`; // Neutral gray
    }

    // Use 0 as the decision boundary
    // Positive scores = red (deceptive), negative scores = green (honest)

    // Apply a scaling factor to map scores to 0-1 range
    // Using tanh-like mapping: most scores are in [-10, 10] range
    const maxAbsScore = 10;
    const normalized = Math.max(-1, Math.min(1, score / maxAbsScore));

    let red, green;

    if (normalized >= 0) {
        // Positive score: interpolate from neutral to red
        red = Math.round(128 + normalized * 127);  // 128 -> 255
        green = Math.round(128 * (1 - normalized));  // 128 -> 0
    } else {
        // Negative score: interpolate from neutral to green
        red = Math.round(128 * (1 + normalized));  // 128 -> 0
        green = Math.round(128 + Math.abs(normalized) * 127);  // 128 -> 255
    }

    return `rgba(${red}, ${green}, 0, ${alpha})`;
}

// Highlight search text
function highlightSearchText(text) {
    if (!currentSearchQuery) return text;

    const regex = new RegExp(`(${escapeRegex(currentSearchQuery)})`, 'gi');
    return text.replace(regex, '<span class="highlight">$1</span>');
}

// Escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Toggle conversation history
function toggleHistory() {
    showHistory = !showHistory;
    const historyDiv = document.getElementById('conversation-history');
    const btn = document.getElementById('toggle-history-btn');

    if (showHistory) {
        historyDiv.style.display = 'block';
        btn.textContent = 'Hide Conversation History';
        displayConversationHistory(currentData[currentIndex]);
    } else {
        historyDiv.style.display = 'none';
        btn.textContent = 'Show Conversation History';
    }
}

// Toggle token scores
function toggleTokens() {
    showTokens = !showTokens;
    const btn = document.getElementById('toggle-tokens-btn');

    if (showTokens) {
        btn.textContent = 'Hide Token Scores';
    } else {
        btn.textContent = 'Show Token Scores';
    }

    displayAssistantResponse(currentData[currentIndex]);
}

// Toggle filters panel
function toggleFilters() {
    const filtersDiv = document.getElementById('filters-container');
    const btn = document.getElementById('toggle-filters-btn');

    if (filtersDiv.style.display === 'none') {
        filtersDiv.style.display = 'block';
        btn.textContent = 'Hide Filters & Sort';
    } else {
        filtersDiv.style.display = 'none';
        btn.textContent = 'Show Filters & Sort';
    }
}

// Apply filters
async function applyFilters() {
    await fetchData();
}

// Perform search
async function performSearch() {
    const searchInput = document.getElementById('search-input');
    currentSearchQuery = searchInput.value.trim();

    if (!currentSearchQuery) {
        showMessage('Please enter a search query', 'warning');
        return;
    }

    await fetchData();
}

// Clear search
async function clearSearch() {
    currentSearchQuery = '';
    document.getElementById('search-input').value = '';
    await fetchData();
}

// Show message to user
function showMessage(message, type = 'info') {
    const messageArea = document.getElementById('message-area');
    messageArea.textContent = message;
    messageArea.className = `message ${type}`;
    messageArea.style.display = 'block';

    setTimeout(() => {
        messageArea.style.display = 'none';
    }, 3000);
}
