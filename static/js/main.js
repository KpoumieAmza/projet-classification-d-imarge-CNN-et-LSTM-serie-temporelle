// ============================================================
// main.js
// Rôle : Logique interactive de l'interface dark mode
//        - Drag & Drop d'images
//        - Appel API Flask /predict
//        - Affichage animé des résultats
//        - Chargement du dashboard
// ============================================================

// ============================================================
// SÉLECTION DES ÉLÉMENTS DOM
// On récupère tous les éléments HTML qu'on va manipuler
// ============================================================

const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const previewZone  = document.getElementById('previewZone');
const previewImg   = document.getElementById('previewImg');
const previewName  = document.getElementById('previewName');
const predictBtn   = document.getElementById('predictBtn');
const resetBtn     = document.getElementById('resetBtn');

// Zones de résultats
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultLoader      = document.getElementById('resultLoader');
const resultContent     = document.getElementById('resultContent');

// Éléments de résultat
const predEmoji    = document.getElementById('predEmoji');
const predClass    = document.getElementById('predClass');
const predConf     = document.getElementById('predConf');
const ringFill     = document.getElementById('ringFill');
const ringValue    = document.getElementById('ringValue');
const processedImg = document.getElementById('processedImg');
const probList     = document.getElementById('probList');
const classBars    = document.getElementById('classBars');

// Fichier sélectionné (stocké globalement pour l'envoi)
let selectedFile = null;

// ============================================================
// NAVIGATION — Lien actif au scroll
// Met en surbrillance le lien de navigation de la section
// actuellement visible dans le viewport
// ============================================================

const navLinks   = document.querySelectorAll('.nav-link');
const sections   = document.querySelectorAll('section[id]');

// IntersectionObserver : détecte quand une section est visible
const navObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      // Retire la classe active de tous les liens
      navLinks.forEach(l => l.classList.remove('active'));
      // Ajoute la classe active au lien correspondant
      const activeLink = document.querySelector(
        `.nav-link[href="#${entry.target.id}"]`
      );
      if (activeLink) activeLink.classList.add('active');
    }
  });
}, {
  // La section est "vue" quand 40% est visible
  threshold: 0.4
});

// On observe chaque section
sections.forEach(s => navObserver.observe(s));

// ============================================================
// DROP ZONE — Clic pour ouvrir le sélecteur de fichier
// ============================================================

dropZone.addEventListener('click', () => {
  // Simule un clic sur l'input file caché
  fileInput.click();
});

// Quand l'utilisateur sélectionne un fichier via le dialogue
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

// ============================================================
// DRAG & DROP — Gestion du glisser-déposer
// ============================================================

// Empêche le comportement par défaut du navigateur
// (ouvrir l'image dans un nouvel onglet)
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  // Ajoute l'effet visuel de survol
  dropZone.classList.add('drag-over');
});

// Retire l'effet quand on quitte la zone
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

// Quand on lâche le fichier sur la zone
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');

  // Récupère le premier fichier déposé
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleFile(file);
  } else {
    showNotification('Veuillez déposer une image valide', 'error');
  }
});

// ============================================================
// FONCTION : handleFile()
// Rôle : Traite le fichier sélectionné
//        - Vérifie que c'est bien une image
//        - Affiche la prévisualisation
//        - Active le bouton de prédiction
// ============================================================

function handleFile(file) {
  // Vérification du type MIME
  if (!file.type.startsWith('image/')) {
    showNotification('Format non supporté — utilisez PNG, JPG ou JPEG', 'error');
    return;
  }

  // Vérification de la taille (max 10 Mo)
  if (file.size > 10 * 1024 * 1024) {
    showNotification('Image trop lourde — maximum 10 Mo', 'error');
    return;
  }

  // Stockage du fichier pour l'envoi ultérieur
  selectedFile = file;

  // Lecture du fichier comme URL de données (base64)
  // pour afficher la prévisualisation sans l'envoyer au serveur
  const reader = new FileReader();

  reader.onload = (e) => {
    // Mise à jour de l'image de prévisualisation
    previewImg.src = e.target.result;

    // Affichage du nom du fichier (tronqué si trop long)
    const name = file.name.length > 30
      ? file.name.substring(0, 27) + '...'
      : file.name;
    previewName.textContent = name;

    // Bascule : cache la drop zone, montre la preview
    dropZone.style.display  = 'none';
    previewZone.style.display = 'block';

    // Active le bouton de prédiction
    predictBtn.disabled = false;

    // Réinitialise les résultats précédents
    showState('placeholder');
  };

  reader.readAsDataURL(file);
}

// ============================================================
// BOUTON RESET — Revenir à l'état initial
// ============================================================

resetBtn.addEventListener('click', () => {
  // Réinitialise le fichier sélectionné
  selectedFile = null;
  fileInput.value = '';

  // Réaffiche la drop zone
  dropZone.style.display    = 'block';
  previewZone.style.display = 'none';

  // Désactive le bouton de prédiction
  predictBtn.disabled = true;

  // Remet le placeholder
  showState('placeholder');
});

// ============================================================
// BOUTON PREDICT — Envoi de l'image au serveur Flask
// ============================================================

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Désactive le bouton pendant la requête
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Analyse...</span>';

  // Affiche le loader
  showState('loader');

  try {
    // --------------------------------------------------------
    // Construction du FormData pour l'envoi multipart
    // C'est le format standard pour envoyer des fichiers en HTTP
    // --------------------------------------------------------
    const formData = new FormData();
    // 'image' doit correspondre au nom dans request.files['image']
    formData.append('image', selectedFile);

    // --------------------------------------------------------
    // Requête POST vers l'API Flask /predict
    // fetch() est l'API moderne pour les requêtes HTTP en JS
    // --------------------------------------------------------
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
      // Pas de Content-Type : le navigateur le génère
      // automatiquement avec le boundary multipart
    });

    // Conversion de la réponse JSON en objet JavaScript
    const data = await response.json();

    // Vérification des erreurs côté serveur
    if (!response.ok || data.error) {
      throw new Error(data.error || 'Erreur serveur');
    }

    // Affichage des résultats
    displayResults(data);
  } catch (error) {
    // En cas d'erreur réseau ou serveur
    showState('placeholder');
    showNotification(`Erreur : ${error.message}`, 'error');
    console.error('[predict] Erreur :', error);
  } finally {
    // Réactive toujours le bouton après la requête
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<span class="btn-icon">⚡</span><span class="btn-text">Lancer la prédiction</span>';
  }
});

// ============================================================
// FONCTION : displayResults()
// Rôle : Affiche les résultats de prédiction dans l'interface
// Args : data — objet JSON retourné par /predict
// ============================================================

function displayResults(data) {
  // --- Prédiction principale ---
  predEmoji.textContent = data.emoji;
  predClass.textContent = data.predicted_class.toUpperCase();
  predConf.textContent  = `Confiance : ${data.confidence_pct}`;

  // --- Jauge circulaire ---
  // stroke-dashoffset contrôle quelle portion du cercle est visible
  // Formule : offset = circonférence × (1 - confidence)
  // Circonférence = 2π × r = 2π × 32 ≈ 201
  const circumference = 201;
  const confidence    = data.confidence;  // Entre 0 et 1
  const offset        = circumference * (1 - confidence);

  // On applique l'offset après un court délai pour déclencher
  // l'animation CSS (la transition ne s'applique pas à 0→0)
  setTimeout(() => {
    ringFill.style.strokeDashoffset = offset;
  }, 100);

  // Valeur textuelle au centre de la jauge
  ringValue.textContent = data.confidence_pct;

  // Couleur de la jauge selon la confiance
  if (confidence >= 0.7) {
    ringFill.style.stroke = 'var(--accent-green)';
  } else if (confidence >= 0.4) {
    ringFill.style.stroke = 'var(--accent-orange)';
  } else {
    ringFill.style.stroke = 'var(--accent-red)';
  }

  // --- Image prétraitée (vue par le CNN en 32×32) ---
  processedImg.src = data.preview_image;

  // --- Barres de probabilités ---
  // On vide la liste précédente
  probList.innerHTML = '';

  // On crée une barre pour chaque classe (déjà triées par proba)
  data.all_predictions.forEach((pred, index) => {
    // Création de l'élément de barre
    const item = document.createElement('div');
    item.className = 'prob-item';

    // Pourcentage de remplissage de la barre
    const pct = (pred.probability * 100).toFixed(1);

    // La première barre (meilleure prédiction) a un style différent
    const isTop = index === 0;

    item.innerHTML = `
      <span class="prob-emoji">${pred.emoji}</span>
      <span class="prob-class">${pred.class}</span>
      <div class="prob-bar-wrap">
        <div class="prob-bar ${isTop ? 'top' : ''}"
             style="width: 0%"
             data-width="${pct}%">
        </div>
      </div>
      <span class="prob-pct">${pred.percentage}</span>
    `;

    probList.appendChild(item);
  });

  // Animation des barres avec un délai échelonné
  // Chaque barre apparaît 60ms après la précédente
  const bars = probList.querySelectorAll('.prob-bar');
  bars.forEach((bar, i) => {
    setTimeout(() => {
      // On lit la largeur cible depuis data-width
      bar.style.width = bar.dataset.width;
    }, 150 + i * 60);
  });

  // Affiche la zone de résultats
  showState('results');
}

// ============================================================
// FONCTION : showState()
// Rôle : Gère l'affichage de la zone de résultats
//        Affiche UN seul état à la fois (placeholder / loader / results)
// ============================================================

function showState(state) {
  // Cache tous les états
  resultPlaceholder.style.display = 'none';
  resultLoader.style.display      = 'none';
  resultContent.style.display     = 'none';

  // Affiche l'état demandé
  if (state === 'placeholder') {
    resultPlaceholder.style.display = 'block';
  } else if (state === 'loader') {
    resultLoader.style.display = 'flex';
  } else if (state === 'results') {
    resultContent.style.display = 'block';
  }
}

// ============================================================
// DASHBOARD — Chargement des barres de précision par classe
// ============================================================

async function loadDashboard() {
  try {
    // Appel à notre route /stats
    const response = await fetch('/stats');
    const data     = await response.json();

    // Construction des barres de précision pour chaque classe
    classBars.innerHTML = '';

    data.class_scores.forEach((item, index) => {
      const bar = document.createElement('div');
      bar.className = 'class-bar-item';

      // Orange/rouge si précision < 80%, cyan/vert sinon
      const isLow = item.precision < 80;

      bar.innerHTML = `
        <span class="class-bar-name">${item.class}</span>
        <div class="class-bar-track">
          <div class="class-bar-fill ${isLow ? 'low' : ''}"
               style="width: 0%"
               data-width="${item.precision}%">
          </div>
        </div>
        <span class="class-bar-pct">${item.precision}%</span>
      `;

      classBars.appendChild(bar);
    });

    // Animation des barres au scroll avec IntersectionObserver
    // Les barres s'animent uniquement quand elles deviennent visibles
    const fillBars = classBars.querySelectorAll('.class-bar-fill');

    const barObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Délai échelonné pour chaque barre
          fillBars.forEach((bar, i) => {
            setTimeout(() => {
              bar.style.width = bar.dataset.width;
            }, i * 80);
          });
          // On arrête d'observer une fois animé
          barObserver.disconnect();
        }
      });
    }, { threshold: 0.3 });

    // On observe le conteneur des barres
    barObserver.observe(classBars);
  } catch (error) {
    console.error('[dashboard] Erreur chargement stats :', error);
  }
}

// ============================================================
// FONCTION : showNotification()
// Rôle : Affiche un toast de notification temporaire
// Args :
//   message — texte à afficher
//   type    — 'success' | 'error' | 'info'
// ============================================================

function showNotification(message, type = 'info') {
  // Supprime la notification précédente si elle existe
  const existing = document.querySelector('.notification');
  if (existing) existing.remove();

  // Crée l'élément de notification
  const notif = document.createElement('div');
  notif.className = 'notification';

  // Couleur selon le type
  const colors = {
    success: 'var(--accent-green)',
    error:   'var(--accent-red)',
    info:    'var(--accent-cyan)'
  };

  // Style inline pour le toast
  notif.style.cssText = `
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 9999;
    padding: 12px 20px;
    background: var(--bg-tertiary);
    border: 1px solid ${colors[type]};
    border-radius: 10px;
    color: ${colors[type]};
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    animation: fadeInUp 0.3s ease;
    max-width: 320px;
  `;

  notif.textContent = message;
  document.body.appendChild(notif);

  // Disparaît automatiquement après 3.5 secondes
  setTimeout(() => {
    notif.style.opacity    = '0';
    notif.style.transition = 'opacity 0.3s ease';
    setTimeout(() => notif.remove(), 300);
  }, 3500);
}

// ============================================================
// INITIALISATION
// Code exécuté au chargement complet de la page
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  // Chargement du dashboard
  loadDashboard();

  console.log('%c DeepCNN Interface ', 'background:#00d4ff;color:#0d0d1a;font-weight:bold;padding:4px 8px;border-radius:4px;');
  console.log('%c Modèle : CustomCNN CIFAR-10 — 84.83% accuracy', 'color:#10b981');
});
// ============================================================
// SECTION LSTM — Logique prédiction boursière
// ============================================================

// ============================================================
// FONCTION : loadLstmChart()
// Rôle : Charge l'historique TSLA et dessine le graphique
//        avec Chart.js (chargé via CDN dans index.html)
// ============================================================

async function loadLstmChart() {
  try {
    // Appel à notre route /lstm_history
    const response = await fetch('/lstm_history');
    const data     = await response.json();

    if (!data.success) throw new Error('Erreur chargement historique');

    // Mise à jour du prix et de la date actuels
    document.getElementById('lstmLastPrice').textContent =
      `$${data.stats.last.toFixed(2)}`;
    document.getElementById('lstmLastDate').textContent =
      data.dates[data.dates.length - 1];

    // Récupération du canvas pour Chart.js
    const ctx = document.getElementById('lstmChart').getContext('2d');

    // Labels : on affiche 1 date sur 10 pour éviter la surcharge
    const labels = data.dates.map((d, i) =>
      i % 10 === 0 ? d.slice(5) : ''   // Format MM-DD
    );

    // Création du graphique Chart.js
    new Chart(ctx, {
      type: 'line',

      data: {
        labels: labels,
        datasets: [{
          label:           'Prix High TSLA ($)',
          data:            data.prices,
          borderColor:     '#00d4ff',       // Ligne cyan
          backgroundColor: 'rgba(0,212,255,0.06)',
          borderWidth:     2,
          pointRadius:     0,               // Pas de points (trop dense)
          pointHoverRadius: 4,
          fill:            true,            // Zone remplie sous la courbe
          tension:         0.3,             // Courbe lissée
        }]
      },

      options: {
        responsive:          true,
        maintainAspectRatio: false,

        // Interaction au survol
        interaction: {
          intersect: false,
          mode:      'index',
        },

        plugins: {
          // Légende cachée (on a notre propre légende HTML)
          legend: { display: false },

          // Infobulle au survol
          tooltip: {
            backgroundColor: '#1a1a35',
            borderColor:     '#2d4a7a',
            borderWidth:     1,
            titleColor:      '#94a3b8',
            bodyColor:       '#00d4ff',
            callbacks: {
              // Format du prix dans l'infobulle
              label: (ctx) => ` $${ctx.parsed.y.toFixed(2)}`
            }
          }
        },

        scales: {
          x: {
            // Axe X : dates
            grid:  { color: 'rgba(30,45,74,0.8)', lineWidth: 0.5 },
            ticks: { color: '#475569', font: { size: 10 } },
          },
          y: {
            // Axe Y : prix en dollars
            grid:  { color: 'rgba(30,45,74,0.8)', lineWidth: 0.5 },
            ticks: {
              color: '#475569',
              font:  { size: 10 },
              // Préfixe $ sur les valeurs de l'axe
              callback: (val) => `$${val.toFixed(0)}`
            }
          }
        }
      }
    });
  } catch (error) {
    console.error('[lstm] Erreur chargement graphique :', error);
  }
}

// ============================================================
// BOUTON PREDICT LSTM
// ============================================================

const lstmPredictBtn = document.getElementById('lstmPredictBtn');
const lstmResult     = document.getElementById('lstmResult');
const lstmLoader     = document.getElementById('lstmLoader');

lstmPredictBtn.addEventListener('click', async () => {
  // Affiche le loader, cache le résultat précédent
  lstmResult.style.display = 'none';
  lstmLoader.style.display = 'flex';
  lstmPredictBtn.disabled  = true;
  lstmPredictBtn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Calcul LSTM...</span>';

  try {
    // Appel à /predict_lstm
    const response = await fetch('/predict_lstm', { method: 'POST' });
    const data     = await response.json();

    if (!data.success) throw new Error(data.error || 'Erreur serveur');

    // Mise à jour du prix prédit
    document.getElementById('lstmPredPrice').textContent =
      `$${data.predicted_price.toFixed(2)}`;

    // Signal hausse / baisse
    const signalEl = document.getElementById('lstmSignal');
    signalEl.textContent  = data.signal;
    signalEl.className    = `lstm-signal ${data.is_up ? 'up' : 'down'}`;

    // Variation en pourcentage
    const sign = data.variation > 0 ? '+' : '';
    document.getElementById('lstmVariation').textContent =
      `Variation estimée : ${sign}${data.variation.toFixed(2)}% vs $${data.last_price.toFixed(2)}`;

    // Cache loader, affiche résultat
    lstmLoader.style.display = 'none';
    lstmResult.style.display = 'block';

    // Notification
    const msg = data.is_up
      ? `Prédiction : hausse vers $${data.predicted_price.toFixed(2)}`
      : `Prédiction : baisse vers $${data.predicted_price.toFixed(2)}`;
    showNotification(msg, data.is_up ? 'success' : 'info');
  } catch (error) {
    lstmLoader.style.display = 'none';
    showNotification(`Erreur LSTM : ${error.message}`, 'error');
    console.error('[lstm] Erreur prédiction :', error);
  } finally {
    // Réactive toujours le bouton
    lstmPredictBtn.disabled  = false;
    lstmPredictBtn.innerHTML = '<span class="btn-icon">⚡</span><span class="btn-text">Prédire le prix de demain</span>';
  }
});

// ============================================================
// Chargement de Chart.js via CDN puis initialisation
// Chart.js doit être chargé AVANT d'appeler loadLstmChart()
// ============================================================

function loadChartJS(callback) {
  // Vérifie si Chart.js est déjà chargé
  if (typeof Chart !== 'undefined') {
    callback();
    return;
  }

  // Chargement dynamique de Chart.js depuis le CDN
  const script  = document.createElement('script');
  script.src    = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js';
  script.onload = callback;   // Lance le callback quand Chart.js est prêt
  document.head.appendChild(script);
}

// Au chargement de la page : charge Chart.js puis le graphique
document.addEventListener('DOMContentLoaded', () => {
  loadChartJS(() => {
    loadLstmChart();
  });
});
