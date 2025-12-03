// API Configuration
const API_BASE_URL = window.location.origin;

// Global state
let currentSearchResults = null;
let sentimentChart = null;
let platformChart = null;
let trendsChart = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
});

// Event Listeners
function initializeEventListeners() {
    const searchForm = document.getElementById('searchForm');
    const trendsBtn = document.getElementById('trendsBtn');
    const llmInsightBtn = document.getElementById('llmInsightBtn');
    
    searchForm.addEventListener('submit', handleSearch);
    
    if (trendsBtn) {
        trendsBtn.addEventListener('click', handleLoadTrends);
    }
    
    if (llmInsightBtn) {
        llmInsightBtn.addEventListener('click', handleLLMInsight);
    }
}

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        console.log('API Health:', data);
        
        if (data.status !== 'healthy' && data.status !== 'degraded') {
            showNotification('Attention: Certains services peuvent être indisponibles', 'warning');
        }
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showNotification('Impossible de se connecter à l\'API', 'error');
    }
}

// Handle Search (Endpoint RAPIDE - Analyse des avis)
async function handleSearch(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const keyword = formData.get('keyword');
    const timeFilter = formData.get('timeFilter');
    const limit = parseInt(formData.get('limit'));
    
    // Get selected platforms
    const platforms = [];
    const platformCheckboxes = document.querySelectorAll('input[name="platform"]:checked');
    platformCheckboxes.forEach(checkbox => platforms.push(checkbox.value));
    
    if (platforms.length === 0) {
        showNotification('Veuillez sélectionner au moins une plateforme', 'warning');
        return;
    }
    
    // Show loading
    showLoading(true);
    hideResults();
    
    try {
        console.log('🚀 Calling FAST endpoint: /api/search (no trends)');
        
        const response = await fetch(`${API_BASE_URL}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                keyword: keyword,
                platforms: platforms,
                limit: limit,
                time_filter: timeFilter,
                include_comments: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentSearchResults = data;
        
        console.log('✅ Fast search completed in', data.execution_time.toFixed(2), 'seconds');
        console.log('Search Results:', data);
        
        // Display results (SANS tendances)
        displayResults(data);
        
        // Afficher les boutons "Analyser les tendances" et "Résumé LLM"
        const trendsBtn = document.getElementById('trendsBtn');
        const llmBtn = document.getElementById('llmInsightBtn');
        if (trendsBtn) {
            trendsBtn.style.display = 'inline-block';
        }
        if (llmBtn) {
            llmBtn.style.display = 'inline-block';
        }
        
        showNotification(`✅ ${data.total_posts} avis analysés en ${data.execution_time.toFixed(1)}s!`, 'success');
        
    } catch (error) {
        console.error('Search Error:', error);
        showNotification('Erreur lors de la recherche. Veuillez réessayer.', 'error');
    } finally {
        showLoading(false);
    }
}

// Display Results (SANS tendances - endpoint rapide)
function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Afficher la section trends (vide pour l'instant)
    document.getElementById('trendsSection').style.display = 'block';
    document.getElementById('trendsPlaceholder').style.display = 'block';
    document.getElementById('trendsContent').style.display = 'none';
    
    // Update stats
    updateStats(data);
    
    // Display filtering statistics
    displayFilteringStats(data.filtering_stats);
    
    // TODO: Réactiver product summary plus tard
    // Display product summary and key points
    // displayProductSummary(data);
    
    // Update charts
    updateSentimentChart(data.overall_sentiment);
    updatePlatformChart(data);
    
    // Display posts
    displayPosts(data.posts);
    
    // TODO: Réactiver aspects plus tard
    // Display aspects if available
    // displayAspects(data.posts);
    
    console.log('✅ Results displayed (without trends)');
}

// Update Stats
function updateStats(data) {
    const sentiment = data.overall_sentiment;
    
    document.getElementById('positiveCount').textContent = `${(sentiment.positive * 100).toFixed(1)}%`;
    document.getElementById('neutralCount').textContent = `${(sentiment.neutral * 100).toFixed(1)}%`;
    document.getElementById('negativeCount').textContent = `${(sentiment.negative * 100).toFixed(1)}%`;
    document.getElementById('totalPosts').textContent = data.total_posts;
}

// ✨ IMPROVED: Display Enhanced Filtering Statistics with Review Metrics
function displayFilteringStats(stats) {
    if (!stats) {
        document.getElementById('filteringStatsSection').style.display = 'none';
        return;
    }
    
    const section = document.getElementById('filteringStatsSection');
    section.style.display = 'block';
    
    // Use new review filtering metrics if available
    const totalFetched = currentSearchResults?.total_fetched_posts || stats.total || 0;
    const reviewsUsed = currentSearchResults?.total_review_posts_used || stats.relevant || 0;
    const filtered = totalFetched - reviewsUsed;
    
    document.getElementById('filterTotal').textContent = totalFetched;
    document.getElementById('filterRelevant').textContent = reviewsUsed;
    document.getElementById('filterFiltered').textContent = filtered;
    document.getElementById('filterAvgScore').textContent = (stats.avg_score || 0).toFixed(2);
    
    // ✨ Show informative message if many posts were filtered
    const filterPercentage = totalFetched > 0 ? (filtered / totalFetched * 100) : 0;
    const filterInfo = section.querySelector('.filter-info');
    
    if (filterPercentage > 50) {
        filterInfo.innerHTML = `
            <i class="fas fa-info-circle"></i> 
            ${filterPercentage.toFixed(0)}% des posts collectés ont été filtrés (questions, photos, etc.). 
            Seuls les vrais avis sont analysés pour une meilleure qualité.
        `;
        filterInfo.style.backgroundColor = '#fff3cd';
        filterInfo.style.color = '#856404';
    } else {
        filterInfo.innerHTML = `
            <i class="fas fa-check-circle"></i> 
            Filtrage qualité appliqué : ${reviewsUsed}/${totalFetched} avis utilisés
        `;
        filterInfo.style.backgroundColor = '#d1ecf1';
        filterInfo.style.color = '#0c5460';
    }
    
    console.log('✨ Enhanced filtering stats:', {
        totalFetched,
        reviewsUsed,
        filtered,
        filterPercentage: filterPercentage.toFixed(1) + '%'
    });
}

// ✨ IMPROVED: Display Product Summary with Enhanced Statistics
function displayProductSummary(data) {
    console.log('✨ Displaying enhanced product summary:', data.product_summary);
    console.log('Key points:', data.key_points);
    
    const summarySection = document.getElementById('productSummarySection');
    const summaryText = document.getElementById('productSummaryText');
    const keyPointsList = document.getElementById('keyPointsList');
    
    if (!summarySection || !summaryText || !keyPointsList) {
        console.error('Product summary elements not found in DOM');
        return;
    }
    
    // Always show section
    summarySection.style.display = 'block';
    
    // ✨ Display product summary (now with statistics + typical phrases)
    if (data.product_summary && data.product_summary.trim() !== '') {
        console.log('Rendering informative aspect-based product summary');
        
        // Parse aspect-based summary (format: "**Aspect**: Description. Details | **Aspect2**: ...")
        const aspects = data.product_summary.split(' | ');
        
        if (aspects.length > 1) {
            // ✨ Multiple aspects detected - display with rich formatting
            const aspectsHTML = aspects.map(aspect => {
                // Parse markdown bold syntax: **Aspect**: Description
                const match = aspect.match(/\*\*([^*]+)\*\*:\s*(.+)/);
                
                if (match) {
                    const name = match[1].trim();
                    const description = match[2].trim();
                    
                    // ✨ Extract sentiment percentage if present
                    const sentimentMatch = description.match(/(\d+)%\s*positive/i);
                    const positivePercent = sentimentMatch ? parseInt(sentimentMatch[1]) : null;
                    
                    // Determine sentiment icon
                    let sentimentEmoji = '😐';
                    if (positivePercent !== null) {
                        if (positivePercent >= 70) sentimentEmoji = '👍';
                        else if (positivePercent <= 30) sentimentEmoji = '👎';
                    }
                    
                    // Determine aspect icon
                    const icon = getAspectIcon(name.toLowerCase());
                    
                    return `
                        <div class="aspect-item enhanced">
                            <div class="aspect-header">
                                <i class="fas fa-${icon}"></i>
                                <strong>${name}</strong>
                                <span class="sentiment-badge">${sentimentEmoji}</span>
                            </div>
                            <div class="aspect-description">
                                ${description}
                            </div>
                        </div>
                    `;
                }
                
                // Fallback: original format "Aspect: Description"
                const [name, ...rest] = aspect.split(': ');
                const description = rest.join(': ');
                
                if (name && description) {
                    const icon = getAspectIcon(name.trim().toLowerCase());
                    return `
                        <div class="aspect-item">
                            <div class="aspect-header">
                                <i class="fas fa-${icon}"></i>
                                <strong>${name}:</strong>
                            </div>
                            <div class="aspect-description">${description}</div>
                        </div>
                    `;
                }
                return '';
            }).join('');
            
            summaryText.innerHTML = `
                <div class="aspect-summary enhanced">
                    ${aspectsHTML}
                </div>
            `;
        } else {
            // Single text - display as before
            summaryText.innerHTML = `
                <div class="summary-content">
                    <i class="fas fa-quote-left"></i>
                    <p>${data.product_summary}</p>
                    <i class="fas fa-quote-right"></i>
                </div>
            `;
        }
    } else {
        console.log('No product summary available');
        summaryText.innerHTML = '<p class="text-muted">Summary not available (English content only)</p>';
    }
    
    // ✨ Display key points with enhanced formatting
    if (data.key_points && data.key_points.length > 0) {
        console.log(`Rendering ${data.key_points.length} key points`);
        
        keyPointsList.innerHTML = data.key_points.map(point => {
            // ✨ Extract sentiment percentages for enhanced display
            const sentimentMatch = point.match(/(\d+)%\s*positive.*?(\d+)%\s*negative.*?(\d+)%\s*neutral/i);
            
            if (sentimentMatch) {
                const pos = parseInt(sentimentMatch[1]);
                const neg = parseInt(sentimentMatch[2]);
                const neu = parseInt(sentimentMatch[3]);
                
                return `
                    <li class="key-point-enhanced">
                        <i class="fas fa-check-circle"></i>
                        <span>${point}</span>
                        <div class="sentiment-bar">
                            <div class="sentiment-segment positive" style="width: ${pos}%" title="${pos}% Positif"></div>
                            <div class="sentiment-segment negative" style="width: ${neg}%" title="${neg}% Négatif"></div>
                            <div class="sentiment-segment neutral" style="width: ${neu}%" title="${neu}% Neutre"></div>
                        </div>
                    </li>
                `;
            }
            
            // Standard key point
            return `
                <li>
                    <i class="fas fa-check-circle"></i>
                    <span>${point}</span>
                </li>
            `;
        }).join('');
    } else {
        console.log('No key points available');
        keyPointsList.innerHTML = '<li class="text-muted">No key points available</li>';
    }
}

// Get icon for specific aspect (handles dynamic aspects)
function getAspectIcon(aspect) {
    const aspectLower = aspect.toLowerCase();
    
    // Common icons for known aspects
    const iconMap = {
        'camera': 'camera',
        'photo': 'camera',
        'picture': 'camera',
        'battery': 'battery-full',
        'power': 'battery-full',
        'screen': 'desktop',
        'display': 'desktop',
        'monitor': 'desktop',
        'price': 'dollar-sign',
        'cost': 'dollar-sign',
        'expensive': 'dollar-sign',
        'affordable': 'dollar-sign',
        'performance': 'tachometer-alt',
        'speed': 'tachometer-alt',
        'fast': 'tachometer-alt',
        'design': 'palette',
        'look': 'palette',
        'style': 'palette',
        'quality': 'award',
        'build': 'award',
        'sound': 'volume-up',
        'audio': 'volume-up',
        'speaker': 'volume-up',
        'music': 'volume-up',
        'service': 'headset',
        'support': 'headset',
        'customer': 'headset',
        'shipping': 'shipping-fast',
        'delivery': 'shipping-fast',
        'size': 'expand',
        'weight': 'weight',
        'color': 'palette',
        'warranty': 'shield-alt',
        'software': 'code',
        'app': 'mobile-alt',
        'storage': 'hdd',
        'memory': 'memory',
        'processor': 'microchip',
        'connectivity': 'wifi',
        'bluetooth': 'bluetooth',
        'charging': 'charging-station',
        'durability': 'shield-alt',
        'comfort': 'couch',
        'ease': 'thumbs-up',
        'value': 'star',
        'packaging': 'box',
        'instructions': 'book',
        'features': 'list-ul'
    };
    
    // Check exact match
    if (iconMap[aspectLower]) {
        return iconMap[aspectLower];
    }
    
    // Check if aspect contains a keyword
    for (const [keyword, icon] of Object.entries(iconMap)) {
        if (aspectLower.includes(keyword)) {
            return icon;
        }
    }
    
    // Default icon for unknown aspects
    return 'tag';
}

// Update Sentiment Chart
function updateSentimentChart(sentiment) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    
    if (sentimentChart) {
        sentimentChart.destroy();
    }
    
    sentimentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positif', 'Neutre', 'Négatif'],
            datasets: [{
                data: [
                    (sentiment.positive * 100).toFixed(1),
                    (sentiment.neutral * 100).toFixed(1),
                    (sentiment.negative * 100).toFixed(1)
                ],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            }
        }
    });
}

// Update Platform Chart
function updatePlatformChart(data) {
    const ctx = document.getElementById('platformChart').getContext('2d');
    
    // Group posts by platform
    const platformData = {};
    data.posts.forEach(post => {
        if (!platformData[post.platform]) {
            platformData[post.platform] = {
                positive: 0,
                neutral: 0,
                negative: 0
            };
        }
        
        if (post.sentiment) {
            platformData[post.platform].positive += post.sentiment.positive;
            platformData[post.platform].neutral += post.sentiment.neutral;
            platformData[post.platform].negative += post.sentiment.negative;
        }
    });
    
    // Calculate averages
    const platforms = Object.keys(platformData);
    const positiveData = [];
    const neutralData = [];
    const negativeData = [];
    
    platforms.forEach(platform => {
        const count = data.posts.filter(p => p.platform === platform).length;
        positiveData.push((platformData[platform].positive / count * 100).toFixed(1));
        neutralData.push((platformData[platform].neutral / count * 100).toFixed(1));
        negativeData.push((platformData[platform].negative / count * 100).toFixed(1));
    });
    
    if (platformChart) {
        platformChart.destroy();
    }
    
    platformChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: platforms.map(p => p.charAt(0).toUpperCase() + p.slice(1)),
            datasets: [
                {
                    label: 'Positif',
                    data: positiveData,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)'
                },
                {
                    label: 'Neutre',
                    data: neutralData,
                    backgroundColor: 'rgba(245, 158, 11, 0.8)'
                },
                {
                    label: 'Négatif',
                    data: negativeData,
                    backgroundColor: 'rgba(239, 68, 68, 0.8)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    stacked: true
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y}%`;
                        }
                    }
                }
            }
        }
    });
}

// Display Posts
function displayPosts(posts) {
    const postsList = document.getElementById('postsList');
    postsList.innerHTML = '';
    
    if (posts.length === 0) {
        postsList.innerHTML = '<p>Aucun post trouvé.</p>';
        return;
    }
    
    posts.forEach(post => {
        const postElement = createPostElement(post);
        postsList.appendChild(postElement);
    });
}

// Create Post Element
function createPostElement(post) {
    const div = document.createElement('div');
    div.className = 'post-item';
    
    const platformBadgeClass = post.platform === 'reddit' ? 'reddit' : 'twitter';
    const platformIcon = post.platform === 'reddit' ? 'fa-reddit' : 'fa-twitter';
    
    let summaryHTML = '';
    if (post.summary) {
        summaryHTML = `
            <div class="post-summary">
                <h4><i class="fas fa-file-alt"></i> Résumé des commentaires</h4>
                <p>${escapeHtml(post.summary)}</p>
            </div>
        `;
    }
    
    let aspectsHTML = '';
    if (post.aspects && post.aspects.length > 0) {
        const aspectBadges = post.aspects.map(aspect => {
            return `
                <span class="aspect-badge ${aspect.sentiment}">
                    ${escapeHtml(aspect.aspect)} (${aspect.mentions})
                </span>
            `;
        }).join('');
        
        aspectsHTML = `
            <div class="post-aspects">
                ${aspectBadges}
            </div>
        `;
    }
    
    const sentiment = post.sentiment || { positive: 0.33, neutral: 0.34, negative: 0.33 };
    
    div.innerHTML = `
        <div class="post-header">
            <div>
                <span class="platform-badge ${platformBadgeClass}">
                    <i class="fab ${platformIcon}"></i> ${post.platform}
                </span>
            </div>
            <div class="post-meta">
                <span><i class="fas fa-user"></i> ${escapeHtml(post.author)}</span>
                <span><i class="fas fa-clock"></i> ${formatDate(post.created_at)}</span>
                <span><i class="fas fa-comment"></i> ${post.num_comments}</span>
            </div>
        </div>
        
        ${post.title ? `<h3 class="post-title">${escapeHtml(post.title)}</h3>` : ''}
        <p class="post-text">${escapeHtml(post.text)}</p>
        
        ${summaryHTML}
        ${aspectsHTML}
        
        <div class="sentiment-bar">
            <div class="sentiment-segment positive" style="width: ${(sentiment.positive * 100).toFixed(1)}%">
                ${(sentiment.positive * 100).toFixed(0)}%
            </div>
            <div class="sentiment-segment neutral" style="width: ${(sentiment.neutral * 100).toFixed(1)}%">
                ${(sentiment.neutral * 100).toFixed(0)}%
            </div>
            <div class="sentiment-segment negative" style="width: ${(sentiment.negative * 100).toFixed(1)}%">
                ${(sentiment.negative * 100).toFixed(0)}%
            </div>
        </div>
    `;
    
    return div;
}

// Display Aspects
// ✨ IMPROVED: Display Aspects with Quality Filtering
function displayAspects(posts) {
    const aspectsSection = document.getElementById('aspectsSection');
    const aspectsList = document.getElementById('aspectsList');
    
    // ✨ Blacklist: mots à ne jamais afficher comme aspects
    const blacklistWords = [
        'thanks', 'thank', 'edit', 'shot', 'update', 'area', 'today', 'yesterday',
        'everyone', 'guys', 'someone', 'anyone', 'people', 'thread', 'post',
        'reddit', 'subreddit', 'op', 'comment', 'comments', 'lol', 'haha'
    ];
    
    // Collect all aspects
    const allAspects = {};
    
    posts.forEach(post => {
        if (post.aspects) {
            post.aspects.forEach(aspect => {
                const aspectLower = aspect.aspect.toLowerCase();
                
                // ✨ Filter: skip blacklisted words
                if (blacklistWords.some(word => aspectLower.includes(word))) {
                    console.log(`Filtering aspect '${aspect.aspect}' (blacklisted)`);
                    return;
                }
                
                // ✨ Filter: skip very short aspects (< 3 chars)
                if (aspect.aspect.length < 3) {
                    return;
                }
                
                if (!allAspects[aspect.aspect]) {
                    allAspects[aspect.aspect] = {
                        mentions: 0,
                        sentiment: aspect.sentiment
                    };
                }
                allAspects[aspect.aspect].mentions += aspect.mentions;
            });
        }
    });
    
    // ✨ Limit to max 8-10 aspects for readability
    const aspectsArray = Object.entries(allAspects)
        .map(([aspect, data]) => ({ aspect, ...data }))
        .sort((a, b) => b.mentions - a.mentions)
        .slice(0, 8);  // ✨ Max 8 aspects (was 10)
    
    if (aspectsArray.length > 0) {
        aspectsSection.style.display = 'block';
        
        // ✨ NEW: Add sentiment emojis for better visualization
        aspectsList.innerHTML = aspectsArray.map(aspect => {
            const sentimentEmoji = aspect.sentiment === 'positive' ? '👍' : 
                                   aspect.sentiment === 'negative' ? '👎' : '😐';
            
            return `
                <span class="aspect-badge ${aspect.sentiment}" title="${aspect.sentiment.toUpperCase()} (${aspect.mentions} mentions)">
                    ${sentimentEmoji} ${escapeHtml(aspect.aspect)} (${aspect.mentions})
                </span>
            `;
        }).join('');
        
        console.log(`✨ Displayed ${aspectsArray.length} quality aspects`);
    } else {
        aspectsSection.style.display = 'none';
    }
}

// Handle Load Trends
// Handle Load Trends (Endpoint LOURD - Analyse de tendances depuis PostgreSQL)
async function handleLoadTrends() {
    if (!currentSearchResults) {
        showNotification('Veuillez d\'abord effectuer une recherche', 'warning');
        return;
    }
    
    const keyword = currentSearchResults.keyword;
    const platforms = currentSearchResults.platforms;
    
    // Obtenir le time_filter depuis le formulaire
    const timeFilterSelect = document.getElementById('timeFilter');
    const timeFilter = timeFilterSelect ? timeFilterSelect.value : 'week';
    
    // Mapper vers time_range pour /api/trends
    const timeRangeMap = {
        'hour': '24h',
        'day': '24h',
        'week': '7d',
        'month': '30d',
        'year': '30d',
        'all': '30d'
    };
    const timeRange = timeRangeMap[timeFilter] || '7d';
    
    // Show loading pour les tendances uniquement
    const trendsLoadingIndicator = document.getElementById('trendsLoadingIndicator');
    const trendsPlaceholder = document.getElementById('trendsPlaceholder');
    const trendsContent = document.getElementById('trendsContent');
    
    if (trendsLoadingIndicator) trendsLoadingIndicator.style.display = 'block';
    if (trendsPlaceholder) trendsPlaceholder.style.display = 'none';
    if (trendsContent) trendsContent.style.display = 'none';
    
    try {
        console.log('📊 Calling HEAVY endpoint: /api/trends (from PostgreSQL)');
        
        const response = await fetch(`${API_BASE_URL}/api/trends`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                keyword: keyword,
                platforms: platforms,
                time_range: timeRange
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log('✅ Trends analysis completed in', data.execution_time.toFixed(2), 'seconds');
        console.log('Trends Data:', data);
        
        // Display trends
        displayTrendsResults(data);
        showNotification(`📊 Tendances analysées en ${data.execution_time.toFixed(1)}s!`, 'success');
        
    } catch (error) {
        console.error('Trends Error:', error);
        showNotification('Erreur lors du chargement des tendances', 'error');
        
        // Réafficher le placeholder en cas d'erreur
        if (trendsPlaceholder) trendsPlaceholder.style.display = 'block';
    } finally {
        if (trendsLoadingIndicator) trendsLoadingIndicator.style.display = 'none';
    }
}

// Display Trends Results (depuis /api/trends)
function displayTrendsResults(data) {
    const trendsPlaceholder = document.getElementById('trendsPlaceholder');
    const trendsContent = document.getElementById('trendsContent');
    
    if (trendsPlaceholder) trendsPlaceholder.style.display = 'none';
    if (trendsContent) trendsContent.style.display = 'block';
    
    // Update header avec keyword
    const trendsHeader = document.querySelector('#trendsSection .section-header h2');
    if (trendsHeader) {
        trendsHeader.innerHTML = `<i class="fas fa-chart-line"></i> Tendances Temporelles - ${data.keyword}`;
    }
    
    // Update trends chart
    if (data.trends && data.trends.length > 0) {
        updateIntelligentTrendsChart(data.trends, data.keyword);
        displayTrendInsights(data.trends);
    } else {
        console.warn('No trends data available');
        showNotification('Aucune donnée de tendance disponible', 'warning');
    }
    
    // Display popular topics
    if (data.popular_topics && data.popular_topics.length > 0) {
        displayPopularTopics(data.popular_topics);
    } else {
        console.warn('No popular topics available');
    }
}

// ✅ DISPLAY INTELLIGENT TRENDS (new integrated function)
function displayIntelligentTrends(trends, keyword) {
    console.log('🚀 Displaying intelligent trends for:', keyword, trends);
    
    document.getElementById('trendsContent').style.display = 'block';
    
    // Update header with keyword
    const trendsHeader = document.querySelector('#trendsSection .section-header h2');
    if (trendsHeader) {
        trendsHeader.innerHTML = `<i class="fas fa-chart-line"></i> Tendances Intelligentes - ${keyword}`;
    }
    
    // Update trends chart with intelligence
    updateIntelligentTrendsChart(trends, keyword);
    
    // Display trend insights
    displayTrendInsights(trends);
}

// Display Trends (legacy - for manual trends loading)
function displayTrends(data) {
    console.log('Displaying trends:', data);
    
    document.getElementById('trendsContent').style.display = 'block';
    
    // Update trends chart
    if (data.trends && data.trends.length > 0) {
        updateTrendsChart(data.trends);
    } else {
        console.warn('No trends data available');
        showNotification('Aucune donnée de tendance disponible', 'warning');
    }
    
    // Display popular topics
    if (data.popular_topics && data.popular_topics.length > 0) {
        displayPopularTopics(data.popular_topics);
    } else {
        console.warn('No popular topics available');
    }
}

// ✅ UPDATE INTELLIGENT TRENDS CHART (enhanced with spikes and growth)
function updateIntelligentTrendsChart(trends, keyword) {
    const ctx = document.getElementById('trendsChart').getContext('2d');
    
    if (trendsChart) {
        trendsChart.destroy();
    }
    
    console.log('🚀 Creating intelligent trends chart for:', keyword, trends);
    
    const datasets = trends.map(trend => {
        const color = trend.platform === 'reddit' ? 'rgb(255, 69, 0)' : 'rgb(29, 161, 242)';
        
        // Main trend line
        const mainDataset = {
            label: `${trend.platform.charAt(0).toUpperCase() + trend.platform.slice(1)} - ${keyword}`,
            data: trend.data_points.map(point => {
                const date = new Date(point.timestamp);
                // 🔧 FIX: Normalize to midnight to avoid Chart.js time grouping issues
                date.setHours(0, 0, 0, 0);
                console.log(`📍 Point: ${date.toISOString().split('T')[0]} = ${point.mentions} mentions`);
                return {
                    x: date,
                    y: point.mentions
                };
            }),
            borderColor: color,
            backgroundColor: color + '33',
            tension: 0.4,
            fill: true,
            pointRadius: trend.data_points.map(point => point.is_spike ? 8 : 4),
            pointBackgroundColor: trend.data_points.map(point => 
                point.is_spike ? 'rgb(255, 0, 0)' : color
            ),
            pointBorderColor: trend.data_points.map(point => 
                point.is_spike ? 'rgb(255, 255, 255)' : color
            ),
            pointBorderWidth: trend.data_points.map(point => point.is_spike ? 3 : 1)
        };
        
        return mainDataset;
    });
    
    console.log('📊 Intelligent chart datasets:', datasets);
    
    // Create intelligent trends chart
    trendsChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    },
                    title: {
                        display: true,
                        text: 'Mentions'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const date = new Date(context[0].parsed.x);
                            return date.toLocaleDateString('fr-FR');
                        },
                        afterBody: function(context) {
                            const pointIndex = context[0].dataIndex;
                            const datasetIndex = context[0].datasetIndex;
                            const trend = trends[datasetIndex];
                            const point = trend.data_points[pointIndex];
                            
                            let info = [];
                            if (point.growth_rate !== null && point.growth_rate !== undefined) {
                                const growthPercent = (point.growth_rate * 100).toFixed(1);
                                info.push(`Croissance: ${growthPercent > 0 ? '+' : ''}${growthPercent}%`);
                            }
                            if (point.is_spike) {
                                info.push('🔥 PIC DÉTECTÉ !');
                            }
                            return info;
                        }
                    }
                }
            }
        }
    });
    
    console.log('✅ Intelligent trends chart created successfully');
}

// Update Trends Chart (legacy)
function updateTrendsChart(trends) {
    const ctx = document.getElementById('trendsChart').getContext('2d');
    
    if (trendsChart) {
        trendsChart.destroy();
    }
    
    console.log('Creating trends chart with data:', trends);
    
    const datasets = trends.map(trend => {
        const color = trend.platform === 'reddit' ? 'rgb(255, 69, 0)' : 'rgb(29, 161, 242)';
        
        return {
            label: trend.platform.charAt(0).toUpperCase() + trend.platform.slice(1),
            data: trend.data_points.map(point => {
                const date = new Date(point.timestamp);
                // 🔧 FIX: Normalize to midnight to avoid Chart.js time grouping issues
                date.setHours(0, 0, 0, 0);
                return {
                    x: date,
                    y: point.mentions
                };
            }),
            borderColor: color,
            backgroundColor: color + '33',
            tension: 0.4,
            fill: true
        };
    });
    
    console.log('Chart datasets:', datasets);
    
    trendsChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    },
                    title: {
                        display: true,
                        text: 'Nombre de mentions'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const date = new Date(context[0].parsed.x);
                            return date.toLocaleDateString('fr-FR');
                        }
                    }
                }
            }
        }
    });
    
    console.log('Trends chart created successfully');
}

// Display Popular Topics
function displayPopularTopicsOld(topics) {
    // Section désactivée
    return;
    
    const topicsList = document.getElementById('topicsList');
    
    if (topics.length === 0) {
        topicsList.innerHTML = '<p>Aucun sujet populaire trouvé.</p>';
        return;
    }
    
    topicsList.innerHTML = topics.map(topic => {
        const sentimentClass = topic.sentiment.dominant;
        const sentimentIcon = sentimentClass === 'positive' ? 'fa-smile' :
                             sentimentClass === 'neutral' ? 'fa-meh' : 'fa-frown';
        
        return `
            <div class="topic-item">
                <h4><i class="fas fa-hashtag"></i> ${escapeHtml(topic.keyword)}</h4>
                <p>
                    <i class="fas ${sentimentIcon}" style="color: var(--${sentimentClass})"></i>
                    ${topic.frequency} mentions - Sentiment ${sentimentClass}
                </p>
            </div>
        `;
    }).join('');
}

// Utility Functions
function showLoading(show) {
    document.getElementById('loadingIndicator').style.display = show ? 'block' : 'none';
}

function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('trendsSection').style.display = 'none';
}

function showNotification(message, type = 'info') {
    // Simple console notification for now
    // You can implement a toast notification system
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Simple alert for important messages
    if (type === 'error') {
        alert(message);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 60) return `Il y a ${minutes} min`;
    if (hours < 24) return `Il y a ${hours}h`;
    if (days < 7) return `Il y a ${days}j`;
    
    return date.toLocaleDateString('fr-FR');
}

// ===== INTELLIGENT TRENDS FUNCTIONS =====

// Display Intelligent Trends (NEW)
function displayIntelligentTrends(trends, keyword) {
    console.log('displayIntelligentTrends called with:', trends, keyword);
    
    if (!trends || trends.length === 0) {
        console.warn('No trends to display');
        return;
    }
    
    document.getElementById('trendsContent').style.display = 'block';
    
    // Update header
    const trendsHeader = document.querySelector('#trendsSection .section-header h2');
    if (trendsHeader) {
        trendsHeader.innerHTML = `<i class="fas fa-chart-line"></i> Tendances Intelligentes - ${keyword}`;
    }
    
    // Update trends chart
    updateIntelligentTrendsChart(trends, keyword);
    
    // Display trend insights
    displayTrendInsights(trends);
}

// Update Intelligent Trends Chart
function updateIntelligentTrendsChart(trends, keyword) {
    const ctx = document.getElementById('trendsChart').getContext('2d');
    
    if (trendsChart) {
        trendsChart.destroy();
    }
    
    console.log('Creating intelligent trends chart for:', keyword, trends);
    
    const datasets = trends.map(trend => {
        const color = trend.platform === 'reddit' ? 'rgb(255, 69, 0)' : 'rgb(29, 161, 242)';
        
        return {
            label: `${trend.platform.charAt(0).toUpperCase() + trend.platform.slice(1)} - ${keyword}`,
            data: trend.data_points.map(point => ({
                x: new Date(point.timestamp),
                y: point.mentions
            })),
            borderColor: color,
            backgroundColor: color + '33',
            tension: 0.4,
            fill: true,
            pointRadius: trend.data_points.map(point => point.is_spike ? 8 : 4),
            pointBackgroundColor: trend.data_points.map(point => 
                point.is_spike ? 'rgb(255, 0, 0)' : color
            ),
            pointBorderColor: trend.data_points.map(point => 
                point.is_spike ? 'rgb(255, 255, 255)' : color
            ),
            pointBorderWidth: trend.data_points.map(point => point.is_spike ? 3 : 1)
        };
    });
    
    trendsChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: { day: 'MMM d' }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: { stepSize: 1 },
                    title: {
                        display: true,
                        text: 'Mentions'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const date = new Date(context[0].parsed.x);
                            return date.toLocaleDateString('fr-FR');
                        },
                        afterBody: function(context) {
                            const pointIndex = context[0].dataIndex;
                            const datasetIndex = context[0].datasetIndex;
                            const trend = trends[datasetIndex];
                            const point = trend.data_points[pointIndex];
                            
                            let info = [];
                            if (point.growth_rate !== null && point.growth_rate !== undefined) {
                                const growthPercent = (point.growth_rate * 100).toFixed(1);
                                info.push(`Croissance: ${growthPercent > 0 ? '+' : ''}${growthPercent}%`);
                            }
                            if (point.is_spike) {
                                info.push('PIC DETECTE !');
                            }
                            return info;
                        }
                    }
                }
            }
        }
    });
    
    console.log('Intelligent trends chart created successfully');
}

// Display Trend Insights
function displayTrendInsights(trends) {
    const trendsContent = document.getElementById('trendsContent');
    
    // Remove existing insights
    const existingInsights = trendsContent.querySelector('.trend-insights');
    if (existingInsights) {
        existingInsights.remove();
    }
    
    // Create insights container
    const insightsContainer = document.createElement('div');
    insightsContainer.className = 'trend-insights';
    insightsContainer.innerHTML = '<h3><i class="fas fa-lightbulb"></i> Insights Intelligents</h3><div class="insights-grid" id="insightsGrid"></div>';
    
    // Insert after chart
    const chartContainer = trendsContent.querySelector('canvas');
    if (chartContainer && chartContainer.parentNode) {
        chartContainer.parentNode.insertBefore(insightsContainer, chartContainer.nextSibling);
    } else {
        trendsContent.appendChild(insightsContainer);
    }
    
    const insightsGrid = document.getElementById('insightsGrid');
    
    // Generate insights for each platform
    trends.forEach(trend => {
        const trendEmoji = trend.trend_direction === 'rising' ? '📈' : 
                          trend.trend_direction === 'falling' ? '📉' : '➡️';
        const sentimentEmoji = trend.sentiment_evolution === 'improving' ? '😊' : 
                              trend.sentiment_evolution === 'declining' ? '😠' : '😐';
        const spikeCount = trend.data_points.filter(point => point.is_spike).length;
        
        const platformCard = document.createElement('div');
        platformCard.className = 'platform-insights';
        platformCard.innerHTML = `
            <div class="insight-card">
                <h4><i class="fab fa-${trend.platform}"></i> ${trend.platform.charAt(0).toUpperCase() + trend.platform.slice(1)}</h4>
                <div class="insight-metrics">
                    <div class="metric">
                        <span class="metric-label">Mentions totales:</span>
                        <span class="metric-value">${trend.total_mentions}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Pic maximum:</span>
                        <span class="metric-value">${trend.peak_mentions || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Tendance:</span>
                        <span class="metric-value">${trendEmoji} ${trend.trend_direction}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Évolution sentiment:</span>
                        <span class="metric-value">${sentimentEmoji} ${trend.sentiment_evolution}</span>
                    </div>
                    ${trend.overall_growth_rate !== null && trend.overall_growth_rate !== undefined ? `
                    <div class="metric">
                        <span class="metric-label">Croissance globale:</span>
                        <span class="metric-value">${(trend.overall_growth_rate * 100).toFixed(1)}%</span>
                    </div>
                    ` : ''}
                    ${spikeCount > 0 ? `
                    <div class="metric spike-alert">
                        <span class="metric-label">Pics détectés:</span>
                        <span class="metric-value">🔥 ${spikeCount}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
        insightsGrid.appendChild(platformCard);
    });
}

// Enhanced Display Popular Topics
function displayPopularTopics(topics) {
    console.log('displayPopularTopics called with:', topics);
    
    // Section désactivée - ne rien afficher
    console.log('Popular topics section is disabled');
    return;
    
    if (!topics || topics.length === 0) {
        console.warn('No topics to display');
        return;
    }
    
    const trendsContent = document.getElementById('trendsContent');
    
    // Remove existing topics
    let topicsContainer = trendsContent.querySelector('.popular-topics-section');
    if (topicsContainer) {
        topicsContainer.remove();
    }
    
    // Create topics container
    topicsContainer = document.createElement('div');
    topicsContainer.className = 'popular-topics-section';
    topicsContainer.innerHTML = '<h3><i class="fas fa-tags"></i> Sujets Populaires</h3><div class="topics-grid" id="topicsGrid"></div>';
    
    // Append to trends content
    trendsContent.appendChild(topicsContainer);
    
    const topicsGrid = document.getElementById('topicsGrid');
    
    // Generate topic cards
    topics.forEach((topic, index) => {
        const sentimentEmoji = topic.sentiment.dominant === 'positive' ? '😊' : 
                              topic.sentiment.dominant === 'negative' ? '😠' : '😐';
        const sentimentColor = topic.sentiment.dominant === 'positive' ? 'success' : 
                               topic.sentiment.dominant === 'negative' ? 'danger' : 'info';
        
        const topicCard = document.createElement('div');
        topicCard.className = 'topic-card';
        topicCard.innerHTML = `
            <div class="topic-rank">${index + 1}</div>
            <div class="topic-content">
                <h4 class="topic-keyword">${topic.keyword}</h4>
                <div class="topic-stats">
                    <span class="topic-frequency">${topic.frequency} mentions</span>
                    <span class="topic-sentiment sentiment-${sentimentColor}">
                        ${sentimentEmoji} ${topic.sentiment.dominant}
                    </span>
                </div>
            </div>
        `;
        topicsGrid.appendChild(topicCard);
    });
}

// 🤖 NEW: Handle LLM Insight Generation
async function handleLLMInsight() {
    console.log('🤖 handleLLMInsight called');
    
    if (!currentSearchResults) {
        showNotification('Veuillez d\'abord effectuer une recherche', 'warning');
        return;
    }
    
    const keyword = currentSearchResults.keyword;
    const platforms = currentSearchResults.platforms;
    
    // Obtenir les dates depuis le timeFilter
    const timeFilterSelect = document.getElementById('timeFilter');
    const timeFilter = timeFilterSelect ? timeFilterSelect.value : 'week';
    
    // Calculer start_date et end_date
    const endDate = new Date();
    let startDate = new Date();
    
    switch(timeFilter) {
        case 'hour':
        case 'day':
            startDate.setDate(startDate.getDate() - 1);
            break;
        case 'week':
            startDate.setDate(startDate.getDate() - 7);
            break;
        case 'month':
            startDate.setMonth(startDate.getMonth() - 1);
            break;
        case 'year':
            startDate.setFullYear(startDate.getFullYear() - 1);
            break;
        default:
            startDate.setDate(startDate.getDate() - 7);
    }
    
    const startDateStr = startDate.toISOString().split('T')[0];
    const endDateStr = endDate.toISOString().split('T')[0];
    
    console.log(`🤖 Generating LLM insight for: ${keyword} (${startDateStr} to ${endDateStr})`);
    
    // S'assurer que la section trends est visible
    // S'assurer que la section results est visible
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
    
    // Show LLM insight section with loading
    const llmSection = document.getElementById('llmInsightSection');
    const llmLoading = document.getElementById('llmInsightLoading');
    const llmContent = document.getElementById('llmInsightContent');
    const llmError = document.getElementById('llmInsightError');
    
    if (!llmSection || !llmLoading || !llmContent || !llmError) {
        console.error('LLM insight elements not found in DOM');
        return;
    }
    
    llmSection.style.display = 'block';
    llmLoading.style.display = 'block';
    llmContent.style.display = 'none';
    llmError.style.display = 'none';
    
    try {
        // Call LLM insight endpoint
        const platformsParam = platforms.map(p => `platforms=${p}`).join('&');
        const url = `${API_BASE_URL}/api/trends/llm-insight?keyword=${encodeURIComponent(keyword)}&start_date=${startDateStr}&end_date=${endDateStr}&${platformsParam}`;
        
        console.log('🤖 Calling LLM endpoint:', url);
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ LLM insight received:', data);
        
        // Display insight
        displayLLMInsight(data);
        
    } catch (error) {
        console.error('❌ LLM insight error:', error);
        
        llmLoading.style.display = 'none';
        llmError.style.display = 'block';
        
        const errorText = document.getElementById('llmInsightErrorText');
        errorText.textContent = `Impossible de générer le résumé LLM : ${error.message}`;
        
        showNotification('Erreur lors de la génération du résumé LLM', 'error');
    }
}

// 🤖 Display LLM Insight
function displayLLMInsight(data) {
    const llmLoading = document.getElementById('llmInsightLoading');
    const llmContent = document.getElementById('llmInsightContent');
    const llmError = document.getElementById('llmInsightError');
    
    llmLoading.style.display = 'none';
    llmError.style.display = 'none';
    llmContent.style.display = 'block';
    
    // Display insight text
    const llmInsightText = document.getElementById('llmInsightText');
    if (llmInsightText) {
        llmInsightText.textContent = data.insight;
    }
    
    // Display stats
    if (data.stats) {
        const stats = data.stats;
        
        document.getElementById('llmStatTotal').textContent = stats.total || 0;
        document.getElementById('llmStatPos').textContent = `${stats.pct_pos || 0}%`;
        document.getElementById('llmStatNeg').textContent = `${stats.pct_neg || 0}%`;
    }
    
    console.log('✅ LLM insight displayed');
    showNotification('Résumé LLM généré avec succès!', 'success');
}
