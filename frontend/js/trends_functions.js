/**
 * 🚀 INTELLIGENT TRENDS FUNCTIONS
 * ==============================
 * 
 * Additional functions for displaying intelligent trend insights
 * and popular topics with enhanced UI
 */

// Display Trend Insights
function displayTrendInsights(trends) {
    console.log('📊 Displaying trend insights for platforms:', trends.length);
    
    const trendsContent = document.getElementById('trendsContent');
    
    // Remove existing insights
    const existingInsights = trendsContent.querySelector('.trend-insights');
    if (existingInsights) {
        existingInsights.remove();
    }
    
    // Create insights container
    const insightsContainer = document.createElement('div');
    insightsContainer.className = 'trend-insights';
    insightsContainer.innerHTML = `
        <h3><i class="fas fa-lightbulb"></i> Insights Intelligents</h3>
        <div class="insights-grid" id="insightsGrid"></div>
    `;
    
    // Insert after chart
    const chartContainer = trendsContent.querySelector('.card');
    chartContainer.parentNode.insertBefore(insightsContainer, chartContainer.nextSibling);
    
    const insightsGrid = document.getElementById('insightsGrid');
    
    // Generate insights for each platform
    trends.forEach(trend => {
        const platformInsights = generatePlatformInsights(trend);
        insightsGrid.appendChild(platformInsights);
    });
}

// Generate Platform Insights
function generatePlatformInsights(trend) {
    const container = document.createElement('div');
    container.className = 'platform-insights';
    
    // Determine trend emoji and color
    const trendEmoji = trend.trend_direction === 'rising' ? '📈' : 
                      trend.trend_direction === 'falling' ? '📉' : '➡️';
    const trendColor = trend.trend_direction === 'rising' ? 'success' : 
                       trend.trend_direction === 'falling' ? 'danger' : 'info';
    
    const sentimentEmoji = trend.sentiment_evolution === 'improving' ? '😊' : 
                          trend.sentiment_evolution === 'declining' ? '😠' : '😐';
    const sentimentColor = trend.sentiment_evolution === 'improving' ? 'success' : 
                           trend.sentiment_evolution === 'declining' ? 'danger' : 'info';
    
    // Count spikes
    const spikeCount = trend.data_points.filter(point => point.is_spike).length;
    
    container.innerHTML = `
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
                <div class="metric trend-${trendColor}">
                    <span class="metric-label">Tendance:</span>
                    <span class="metric-value">${trendEmoji} ${trend.trend_direction}</span>
                </div>
                <div class="metric sentiment-${sentimentColor}">
                    <span class="metric-label">Évolution sentiment:</span>
                    <span class="metric-value">${sentimentEmoji} ${trend.sentiment_evolution}</span>
                </div>
                ${trend.overall_growth_rate !== null && trend.overall_growth_rate !== undefined ? `
                <div class="metric growth-${trend.overall_growth_rate >= 0 ? 'positive' : 'negative'}">
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
    
    return container;
}

// Enhanced Display Popular Topics
function displayPopularTopics(topics) {
    console.log('🏷️ Popular topics section disabled - not displaying');
    
    // Section désactivée - ne rien afficher
    return;
    
    const trendsContent = document.getElementById('trendsContent');
    
    // Remove existing topics
    const existingTopics = trendsContent.querySelector('.popular-topics-section');
    if (existingTopics) {
        existingTopics.remove();
    }
    
    // Create topics container
    const topicsContainer = document.createElement('div');
    topicsContainer.className = 'popular-topics-section';
    topicsContainer.innerHTML = `
        <h3><i class="fas fa-tags"></i> Sujets Populaires</h3>
        <div class="topics-grid" id="topicsGrid"></div>
    `;
    
    // Append to trends content
    trendsContent.appendChild(topicsContainer);
    
    const topicsGrid = document.getElementById('topicsGrid');
    
    // Generate topic cards
    topics.forEach((topic, index) => {
        const topicCard = createTopicCard(topic, index + 1);
        topicsGrid.appendChild(topicCard);
    });
}

// Create Topic Card
function createTopicCard(topic, rank) {
    const container = document.createElement('div');
    container.className = 'topic-card';
    
    const sentimentEmoji = topic.sentiment.dominant === 'positive' ? '😊' : 
                          topic.sentiment.dominant === 'negative' ? '😠' : '😐';
    const sentimentColor = topic.sentiment.dominant === 'positive' ? 'success' : 
                           topic.sentiment.dominant === 'negative' ? 'danger' : 'info';
    
    container.innerHTML = `
        <div class="topic-rank">${rank}</div>
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
    
    return container;
}

// Remove manual trends button functionality (since trends are now automatic)
function removeManualTrendsButton() {
    const loadTrendsBtn = document.getElementById('loadTrendsBtn');
    if (loadTrendsBtn) {
        loadTrendsBtn.style.display = 'none';
        console.log('✅ Manual trends button hidden (trends now automatic)');
    }
}

// Initialize automatic trends
document.addEventListener('DOMContentLoaded', function() {
    // Hide manual trends button since trends are now automatic
    removeManualTrendsButton();
    
    console.log('🚀 Intelligent trends system initialized');
});
