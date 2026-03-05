const examples = {
  1: "This film is absolutely beautiful, emotionally rich, and deeply human.",
  2: "The product is horrible, poorly designed, and a complete waste of money.",
  3: "The idea is interesting but the execution feels weak and rushed.",
  4: "Yeah, great job — nothing works exactly as advertised.",
  10: "The movie has strong performances, though the pacing is uneven."
};

function useExample(id) {
  const textInput = document.getElementById("textInput");
  textInput.value = examples[id];
  textInput.focus();
}

async function analyze() {
  const text = document.getElementById("textInput").value;
  const sentimentBox = document.getElementById("sentimentBox");
  const wordCountBox = document.getElementById("wordCountBox");
  const tokensDiv = document.getElementById("tokens");
  const barChartDiv = document.getElementById("barChart");

  const btnText = document.querySelector(".btn-text");
  const loader = document.getElementById("loader");
  const analyzeBtn = document.getElementById("analyzeBtn");

  if (!text.trim()) {
    alert("Please enter a data sequence to analyze.");
    return;
  }

  // UI loading state
  sentimentBox.innerHTML = "Analyzing...";
  sentimentBox.className = "metric-value neutral";
  wordCountBox.innerHTML = "—";

  tokensDiv.innerHTML = "";
  barChartDiv.innerHTML = "";

  btnText.style.display = "none";
  loader.style.display = "block";
  analyzeBtn.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (data.error) {
      alert(data.error);
      sentimentBox.innerHTML = "Error";
      return;
    }

    // 1. Update Metrics Dashboard
    let sentimentCls = "neutral";
    if (data.sentiment === "Positive") sentimentCls = "positive";
    if (data.sentiment === "Negative") sentimentCls = "negative";

    sentimentBox.innerHTML = data.sentiment;
    sentimentBox.className = `metric-value ${sentimentCls}`;

    wordCountBox.innerHTML = `${data.word_count} W / ${data.char_count} C`;
    wordCountBox.className = "metric-value neutral";

    // 2. Render Token Heatmap & Gather Chart Data
    if (!data.explanation || data.explanation.length === 0) {
      tokensDiv.innerHTML = "<div class='placeholder-text'>No strong contributing tokens detected.</div>";
      barChartDiv.innerHTML = "<div class='placeholder-text'>No data to chart.</div>";
      return;
    }

    // Convert to flat array and sort by importance for chart
    let chartData = [];

    data.explanation.forEach(([token, polarity, score]) => {
      // Add token chip to heatmap
      const span = document.createElement("span");
      span.className = "token";

      // Calculate colors based on sentiment and score intensity
      const intensity = Math.min(score * 1.5, 1);

      let rgb = "99, 102, 241";
      if (polarity === "POS" || (polarity === "NEUTRAL" && data.sentiment === "Positive")) {
        rgb = "16, 185, 129"; // Green
      } else if (polarity === "NEG" || (polarity === "NEUTRAL" && data.sentiment === "Negative")) {
        rgb = "239, 68, 68"; // Red
      } else {
        rgb = "148, 163, 184"; // Slate
      }

      span.style.background = `rgba(${rgb}, ${intensity * 0.4})`;
      span.style.border = `1px solid rgba(${rgb}, ${Math.max(intensity, 0.2)})`;
      span.style.color = intensity > 0.6 ? "#fff" : "var(--text-primary)";

      span.innerText = token;

      const formattedScore = (score * 100).toFixed(1) + "%";
      span.setAttribute(
        "data-tooltip",
        `Aspect: ${polarity} | Importance: ${formattedScore}`
      );

      tokensDiv.appendChild(span);

      // Collect for chart
      chartData.push({
        token: token,
        polarity: polarity,
        score: score,
        rgb: rgb
      });
    });

    // 3. Render Top Features Chart
    chartData.sort((a, b) => b.score - a.score);
    const topFeatures = chartData.slice(0, 5); // Take top 5

    if (topFeatures.length === 0) {
      barChartDiv.innerHTML = "<div class='placeholder-text'>No significant features.</div>";
    } else {
      topFeatures.forEach((item) => {
        const row = document.createElement("div");
        row.className = "bar-row";

        // Normalize width based on max score in this set
        const maxScore = topFeatures[0].score;
        const widthPercent = Math.max((item.score / maxScore) * 100, 2); // Min 2% width

        row.innerHTML = `
          <div class="bar-label" title="${item.token}">${item.token}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${widthPercent}%; background: rgb(${item.rgb});"></div>
          </div>
          <div class="bar-value">${(item.score * 100).toFixed(1)}%</div>
        `;

        barChartDiv.appendChild(row);
      });
    }

  } catch (error) {
    console.error("Error during analysis:", error);
    alert("An error occurred while connecting to the server.");
    sentimentBox.innerHTML = "Error";
  } finally {
    // Reset loading state
    btnText.style.display = "block";
    loader.style.display = "none";
    analyzeBtn.disabled = false;
  }
}
