
    function normalizePath(p) {
      if (!p) return "";
      let s = String(p).replace(/\\/g, '/');
      const idx = s.indexOf('output/');
      if (idx !== -1) return s.substring(idx);
      const pIdx = s.indexOf('personajes/');
      if (pIdx !== -1) return s.substring(pIdx);
      const mIdx = s.indexOf('dataset_moldes/');
      if (mIdx !== -1) return s.substring(mIdx);
      return s;
    }

    let appState = {
      characters: [],
      selectedChar: null,
      selectedFront: null,
      format: "8x12",
      poseMap: [],
      selectedCell: 0,
      animPlaying: true,
      animFps: 8,
      animZoom: 4,
      animCycle: "caminar_sur",
      animFrameIndex: 0,
      animTimer: null,
      showGridLines: true,
      zoomLevel: 1.0,
      activeRunDir: null,
      forgeConnected: false
    };

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
    }

    async function initApp() {
      await fetchStatus();
      await fetchCharacters();
      await fetchPoseMap("8x12");
      buildGridCells();
      setGridZoom(0.6);
      startStatusPolling();
      startAnimationLoop();
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        document.getElementById('gpuBadge').textContent = `💻 GPU: ${data.gpu_name} (${data.vram_gb} GB)`;
        
        const tBadge = document.getElementById('trainingBadge');
        const warnBanner = document.getElementById('gpuWarningBanner');
        updateMonitorTab(data.training_info);
        if (data.training_active) {
          tBadge.className = "badge train-active";
          const ep = data.training_info.epoch || '?';
          tBadge.textContent = `⚙️ Entrenamiento Activo (Época ${ep})`;
          warnBanner.style.display = 'block';
        } else {
          tBadge.className = "badge train-idle";
          tBadge.textContent = "⚪ GPU Libre para Inferencia";
          warnBanner.style.display = 'none';
        }

        appState.forgeConnected = !!data.forge_connected;
        const fBadge = document.getElementById('forgeBadge');
        if (data.forge_connected) {
          fBadge.className = "badge forge-on";
          fBadge.textContent = `🟢 Forge SD1.5 Activo`;
        } else {
          fBadge.className = "badge forge-off";
          fBadge.textContent = "⚪ Forge Desconectado";
        }

        populateCheckpoints(data.checkpoints);

        if (data.active_job && data.active_job.status === "running") {
          document.getElementById('progressGroup').style.display = 'flex';
          document.getElementById('progressBar').style.background = 'var(--accent)';
          document.getElementById('progressBar').style.width = data.active_job.progress + '%';
          document.getElementById('progressText').textContent = `${data.active_job.progress}% — ${data.active_job.message}`;
        } else if (data.active_job && data.active_job.status === "completed") {
          document.getElementById('progressBar').style.background = 'var(--success)';
          document.getElementById('progressBar').style.width = '100%';
          document.getElementById('progressText').textContent = '✅ ' + (data.active_job.message || 'Completado');
          if (data.active_job.run_dir && appState.activeRunDir !== data.active_job.run_dir) {
            loadRunSheet(data.active_job.run_dir);
          }
        } else if (data.active_job && data.active_job.status === "error") {
          document.getElementById('progressGroup').style.display = 'flex';
          document.getElementById('progressBar').style.background = 'var(--danger)';
          document.getElementById('progressBar').style.width = '100%';
          document.getElementById('progressText').textContent = '❌ ' + (data.active_job.error || 'Error');
        }
      } catch (e) {
        console.error("Error al obtener estado:", e);
      }
    }

    let lastCkptHash = "";
    function populateCheckpoints(ckpts) {
      if (!ckpts) return;
      const list = ckpts[appState.format] || [];
      const hash = appState.format + ":" + list.map(c => c.filename).join(",");
      if (hash === lastCkptHash) return;
      lastCkptHash = hash;

      const sel = document.getElementById('ckptSelect');
      const prevVal = sel.value;
      sel.innerHTML = '';
      if (list.length === 0) {
        const fallback = ckpts['16x4'] || [];
        fallback.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.path;
          opt.textContent = `${c.filename} (${c.size_mb} MB) [Base Transfer]`;
          sel.appendChild(opt);
        });
        return;
      }
      list.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.path;
        opt.textContent = `${c.filename} (${c.size_mb} MB)`;
        if (c.path === prevVal || c.is_best) opt.selected = true;
        sel.appendChild(opt);
      });
    }

    async function fetchCharacters() {
      try {
        const res = await fetch('/api/characters');
        appState.characters = await res.json();
        const sel = document.getElementById('charSelect');
        sel.innerHTML = '';
        appState.characters.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.name;
          opt.textContent = `${c.name} ${c.has_ground_truth ? '★' : ''}`;
          sel.appendChild(opt);
        });
        if (appState.characters.length > 0) {
          sel.value = appState.characters[0].name;
          onCharacterChange();
        }
      } catch (e) {
        console.error("Error al cargar personajes:", e);
      }
    }

    function onCharacterChange() {
      const name = document.getElementById('charSelect').value;
      const char = appState.characters.find(c => c.name === name);
      if (!char) return;
      appState.selectedChar = char;
      const fSel = document.getElementById('frontSelect');
      fSel.innerHTML = '';
      char.fronts.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.path;
        opt.textContent = f.filename;
        fSel.appendChild(opt);
      });
      if (char.fronts.length > 0) {
        fSel.value = char.fronts[0].path;
        onFrontChange();
      }
    }

    function onFrontChange() {
      const fPath = document.getElementById('frontSelect').value;
      appState.selectedFront = fPath;
      const thumb = document.getElementById('frontThumb');
      thumb.src = fPath;
      document.getElementById('frontInfo').textContent = fPath.split('/').pop();
    }

        function onEngineChange() {
      const eng = document.getElementById('engineSelect').value;
      const ckptLabel = document.getElementById('ckptLabel');
      if (eng === 'forge') {
        if (ckptLabel) ckptLabel.style.display = 'none';
        if (!appState.forgeConnected) {
          alert("ℹ️ Has seleccionado Stable Diffusion Forge.\nActualmente Forge figura DESCONECTADO en http://127.0.0.1:7860.\n\nSi deseas generar con Forge, inicia el WebUI de Forge.\nSi deseas generar de inmediato de forma 100% local, selecciona 'PyTorch UNet (Motor Local Propio)'.");
        }
      } else {
        if (ckptLabel) ckptLabel.style.display = 'flex';
      }
    }

    function onFormatChange() {
      appState.format = document.getElementById('formatSelect').value;
      const box = document.getElementById('sheetViewBox');
      if (appState.format === "16x4") {
        box.classList.add('format-16x4');
      } else {
        box.classList.remove('format-16x4');
      }
      document.getElementById('sheetFormatBadge').textContent = appState.format === "8x12" ? "8x12 (96 Celdas)" : "16x4 (64 Celdas)";
      fetchPoseMap(appState.format).then(buildGridCells);
      fetchStatus();
    }

    async function fetchPoseMap(format) {
      try {
        const res = await fetch('/api/pose_map?format=' + format);
        const data = await res.json();
        appState.poseMap = data.frames;
      } catch (e) {
        console.error("Error al cargar pose map:", e);
      }
    }

    function buildGridCells() {
      const overlay = document.getElementById('gridOverlay');
      overlay.innerHTML = '';
      const cols = appState.format === "8x12" ? 8 : 4;
      const rows = appState.format === "8x12" ? 12 : 16;
      overlay.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
      overlay.style.gridTemplateRows = `repeat(${rows}, 1fr)`;

      const total = cols * rows;
      for (let i = 0; i < total; i++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell' + (i === appState.selectedCell ? ' selected' : '');
        cell.dataset.index = i;
        cell.onclick = () => selectCell(i);

        const badge = document.createElement('div');
        badge.className = 'cell-badge';
        badge.textContent = '#' + (i < 10 ? '0' + i : i);
        cell.appendChild(badge);

        overlay.appendChild(cell);
      }
      selectCell(appState.selectedCell);
    }

    function selectCell(idx) {
      appState.selectedCell = idx;
      document.querySelectorAll('.grid-cell').forEach(c => {
        c.classList.toggle('selected', parseInt(c.dataset.index) === idx);
      });
      const info = appState.poseMap[idx];
      if (info) {
        document.getElementById('inspIndex').textContent = `#${idx < 10 ? '0'+idx : idx} (Frame ${idx+1})`;
        document.getElementById('inspRowCol').textContent = `Fila ${info.row + 1}, Col ${info.col + 1}`;
        document.getElementById('inspDir').textContent = info.direction.toUpperCase();
        document.getElementById('inspAction').textContent = info.action_desc;
        document.getElementById('inspPhase').textContent = info.sub_phase;
      }
      updateSelectedFramePreview(idx);
    }

    function updateSelectedFramePreview(idx) {
      const img = document.getElementById('selectedFrameImg');
      const normDir = normalizePath(appState.activeRunDir);
      if (normDir) {
        const fStr = idx < 10 ? '00' + idx : (idx < 100 ? '0' + idx : idx);
        img.src = `${normDir}/enhanced_frames/frame_${fStr}.png?t=` + Date.now();
      } else {
        const fStr = (idx+1 < 10 ? '00'+(idx+1) : (idx+1 < 100 ? '0'+(idx+1) : idx+1));
        img.src = `dataset_moldes/molde_8x12_96_poses/pose_${fStr}_r01_c01.png`;
      }
    }

    function toggleGridLines() {
      appState.showGridLines = !appState.showGridLines;
      document.querySelectorAll('.grid-cell').forEach(c => {
        c.style.borderColor = appState.showGridLines ? 'rgba(255,255,255,0.08)' : 'transparent';
      });
    }

    function zoomGrid(factor) {
      setGridZoom(appState.zoomLevel * factor);
    }
    function setGridZoom(lvl) {
      appState.zoomLevel = Math.max(0.3, Math.min(2.5, lvl));
      document.getElementById('sheetViewBox').style.transform = `scale(${appState.zoomLevel})`;
      document.getElementById('sheetViewBox').style.transformOrigin = 'top center';
    }

    async function startGeneration(mode) {
      const engine = document.getElementById('engineSelect').value;
      if (engine === 'forge' && !appState.forgeConnected) {
        alert("⚠️ El motor Forge SD1.5 no está disponible porque figura desconectado en http://127.0.0.1:7860.\n\nPor favor cambia el motor a 'PyTorch UNet (Motor Local Propio)' para generar con tu modelo local.");
        return;
      }
      const charSel = document.getElementById('charSelect');
      const charName = (appState.selectedChar && appState.selectedChar.name) || (charSel ? charSel.value : '');
      if (!charName) {
        alert("⚠️ Por favor selecciona un personaje de la lista.");
        return;
      }

      const frontSel = document.getElementById('frontSelect');
      const frontPath = appState.selectedFront || (frontSel ? frontSel.value : '');
      if (!frontPath) {
        alert("⚠️ Por favor selecciona una ilustración frontal.");
        return;
      }

      const ckptSel = document.getElementById('ckptSelect');
      const ckptPath = ckptSel ? ckptSel.value : '';
      if (engine === 'pytorch' && !ckptPath) {
        alert("⚠️ No hay ningún checkpoint seleccionado para este formato. Se requiere un archivo .pt en la carpeta checkpoints.");
        return;
      }

      const btn4 = document.getElementById('btnTest4');
      const btnFull = document.getElementById('btnFullSheet');
      const btnResume = document.getElementById('btnResume');
      if (mode === 'test_4poses' && btn4) btn4.textContent = "⏳ Iniciando 4 Poses...";
      if (mode === 'full' && btnFull) btnFull.textContent = "⏳ Iniciando 96 Frames...";
      if (mode === 'resume' && btnResume) btnResume.textContent = "⏳ Reanudando...";

      const payload = {
        character: charName,
        front_image: frontPath,
        format: appState.format,
        engine: engine,
        checkpoint: ckptPath,
        mode: mode,
        run_dir: mode === 'resume' ? appState.activeRunDir : null
      };

      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.status === 409) {
          alert("⚠️ " + (data.error || "Ya hay un trabajo de generación en curso."));
        } else if (data.status === "gpu_busy") {
          alert("⚠️ " + data.message);
        } else if (!res.ok) {
          alert("⚠️ Error del servidor: " + (data.error || res.statusText));
        } else {
          document.getElementById('progressGroup').style.display = 'flex';
          document.getElementById('progressBar').style.background = 'var(--accent)';
          document.getElementById('progressBar').style.width = '2%';
          document.getElementById('progressText').textContent = 'Iniciando generación...';
        }
      } catch (e) {
        alert("Error de conexión al iniciar generación: " + e);
      } finally {
        setTimeout(() => {
          if (btn4) btn4.textContent = "🧪 Generar 4 Poses de Prueba";
          if (btnFull) btnFull.textContent = "🚀 Generar Hoja Completa (96 Frames)";
          if (btnResume) btnResume.textContent = "🔄 Reanudar Trabajo Incompleto";
        }, 1500);
      }
    }

    async function regenerateSelectedFrame() {
      if (!appState.activeRunDir) {
        alert("Genera primero una hoja o carga un trabajo para poder regenerar un frame.");
        return;
      }
      const payload = {
        frame_idx: appState.selectedCell,
        run_dir: appState.activeRunDir,
        character: (appState.selectedChar && appState.selectedChar.name) || document.getElementById("charSelect").value,
        front_image: appState.selectedFront,
        format: appState.format
      };
      try {
        await fetch('/api/regenerate_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        setTimeout(() => {
          loadRunSheet(appState.activeRunDir);
          selectCell(appState.selectedCell);
        }, 1200);
      } catch (e) {
        alert("Error al regenerar frame: " + e);
      }
    }

    function loadRunSheet(runDir) {
      const normDir = normalizePath(runDir);
      appState.activeRunDir = normDir;
      const cleanImg = `${normDir}/spritesheet_clean.png?t=` + Date.now();
      document.getElementById('activeSheetImg').src = cleanImg;
      document.getElementById('activeRunBadge').textContent = normDir.split('/').pop();
      updateSelectedFramePreview(appState.selectedCell);
      auditCurrentRun(normDir);
    }

    async function auditCurrentRun(runDir) {
      try {
        const res = await fetch(`/api/quality_audit?run_dir=${encodeURIComponent(runDir)}&format=${appState.format}`);
        const data = await res.json();
        document.getElementById('qcCompleteness').textContent = data.completeness_score + '%';
        document.getElementById('qcFramesCount').textContent = `${data.valid_frames_count} / ${data.total_frames_expected} frames`;
        document.getElementById('qcAlpha').textContent = data.alpha_purity_score + '%';
        document.getElementById('qcMargins').textContent = data.border_safety_score + '%';

        let rText = `Auditoría del Spritesheet [${runDir}]:\n`;
        rText += `• Estado: ${data.is_ready_for_game ? '✅ 100% LISTO PARA UNITY' : '⚠️ BORRADOR EN PROCESO'}\n`;
        rText += `• Celdas vacías: ${data.empty_cells.length} frames\n`;
        rText += `• Frames con contacto en bordes: ${data.border_touch_cells.length}\n`;
        rText += `• Puntuación de pureza de alfa: ${data.alpha_purity_score}%\n`;
        document.getElementById('qcReportText').textContent = rText;
      } catch (e) {
        console.error("Error audit:", e);
      }
    }

    function downloadCleanPng() {
      if (!appState.activeRunDir) return alert("No hay hoja cargada para exportar.");
      window.open(`${appState.activeRunDir}/spritesheet_clean.png`, '_blank');
    }
    function downloadGridPng() {
      if (!appState.activeRunDir) return alert("No hay hoja cargada para exportar.");
      window.open(`${appState.activeRunDir}/spritesheet_grid.png`, '_blank');
    }
    function downloadUnityJson() {
      if (!appState.activeRunDir) return alert("No hay hoja cargada para exportar.");
      window.open(`${appState.activeRunDir}/unity_metadata.json`, '_blank');
    }

    function startAnimationLoop() {
      const canvas = document.getElementById('animCanvas');
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;

      function render() {
        if (!appState.animPlaying) return;
        drawAnimFrame(ctx);
        appState.animFrameIndex++;
      }
      appState.animTimer = setInterval(render, 1000 / appState.animFps);
    }

    function drawAnimFrame(ctx) {
      const cfg = {
        "caminar_sur": [0,1,2,3,4,5,6,7],
        "caminar_sureste": [8,9,10,11,12,13,14,15],
        "caminar_este": [16,17,18,19,20,21,22,23],
        "caminar_noreste": [24,25,26,27,28,29,30,31],
        "caminar_norte": [32,33,34,35,36,37,38,39],
        "caminar_noroeste": [40,41,42,43,44,45,46,47],
        "caminar_oeste": [48,49,50,51,52,53,54,55],
        "caminar_suroeste": [56,57,58,59,60,61,62,63],
        "cocinar_bowl": [64,65,66,67,68,69,70,71],
        "cocinar_estacion": [72,73,74,75,76,77,78,79],
        "pensar": [80,81,82,83],
        "cargar_caja": [84,85,86,87],
        "servir_plato": [88,89,90,91],
        "celebrar": [92,93,94,95],
        "idle_reposo": [0, 80]
      };

      const frames = cfg[appState.animCycle] || [0];
      const fIdx = frames[appState.animFrameIndex % frames.length];
      document.getElementById('animFrameLabel').textContent = `Frame: #${fIdx < 10 ? '0'+fIdx : fIdx}`;

      const sheetImg = document.getElementById('activeSheetImg');
      if (sheetImg && sheetImg.complete && sheetImg.naturalWidth > 0) {
        const cols = appState.format === "8x12" ? 8 : 4;
        const rows = appState.format === "8x12" ? 12 : 16;
        const cw = sheetImg.naturalWidth / cols;
        const ch = sheetImg.naturalHeight / rows;
        const r = Math.floor(fIdx / cols);
        const c = fIdx % cols;
        const sx = c * cw;
        const sy = r * ch;

        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        const zoom = appState.animZoom;
        const dw = cw * zoom;
        const dh = ch * zoom;
        const dx = (ctx.canvas.width - dw) / 2;
        const dy = (ctx.canvas.height - dh) / 2;
        ctx.drawImage(sheetImg, sx, sy, cw, ch, dx, dy, dw, dh);
      }
    }

    function togglePlayPause() {
      appState.animPlaying = !appState.animPlaying;
      document.getElementById('playBtn').textContent = appState.animPlaying ? '⏸️ Pausar' : '▶️ Reproducir';
    }
    function onAnimationCycleChange() {
      appState.animCycle = document.getElementById('animCycleSelect').value;
      appState.animFrameIndex = 0;
    }
    function onFpsChange(val) {
      appState.animFps = parseInt(val);
      document.getElementById('fpsVal').textContent = val + ' FPS';
      clearInterval(appState.animTimer);
      startAnimationLoop();
    }
    function onAnimZoomChange(val) {
      appState.animZoom = parseInt(val);
      document.getElementById('zoomVal').textContent = val + 'x';
    }

    function updateMonitorTab(info) {
      if (!info || Object.keys(info).length === 0) return;
      document.getElementById('monEpoch').textContent = `${info.epoch || '--'} / ${info.total_epochs || 150}`;
      const elapsed = info.elapsed_sec ? Math.round(info.elapsed_sec) + 's' : (info.elapsed_s ? Math.round(info.elapsed_s) + 's' : '--');
      document.getElementById('monTime').textContent = `Tiempo transcurrido: ${elapsed}`;
      document.getElementById('monGloss').textContent = info.g_loss ? Number(info.g_loss).toFixed(4) : '--';
      const l1_str = info.l1_loss ? Number(info.l1_loss).toFixed(4) : (info.l1 ? Number(info.l1).toFixed(4) : '--');
      const edge_str = info.edge_loss ? Number(info.edge_loss).toFixed(4) : (info.edge ? Number(info.edge).toFixed(4) : '--');
      document.getElementById('monL1').textContent = `Color L1: ${l1_str} | Bordes: ${edge_str}`;
      document.getElementById('monTemp').textContent = `GPU: ${info.gpu_temp || '--'}°C`;

      const tBadge = document.getElementById('trainStatusBadge');
      if (tBadge) {
        if (info.status === 'ENTRENANDO') {
          tBadge.className = 'badge train-active';
          tBadge.textContent = `⚙️ Aprendiendo (Época ${info.epoch})`;
        } else if (info.status === 'COMPLETADO') {
          tBadge.className = 'badge forge-on';
          tBadge.textContent = '✅ Modelo Entrenado y Listo';
        } else {
          tBadge.className = 'badge';
          tBadge.textContent = '⚪ En Pausa';
        }
      }

      // Actualizar imagenes de comparativa visual en vivo
      const pImg = document.getElementById('monPreviewImg');
      if (pImg) pImg.src = 'training_samples/latest_preview.png?t=' + Date.now();
      const cImg = document.getElementById('monCompImg');
      if (cImg) {
        cImg.style.display = 'block';
        cImg.src = 'training_samples/latest_detail_comparison.png?t=' + Date.now();
      }

      const lBox = document.getElementById('trainLogBox');
      if (lBox) {
        lBox.textContent = `[${info.timestamp || ''}] Fase: ${info.phase || 'Supervisado'} | G_Loss: ${info.g_loss || '--'} | D_Loss: ${info.d_loss || '--'} | Frames Ground-Truth: ${info.total_frames || 1008}`;
      }
    }

    async function startSupervisedTraining() {
      const epochs = document.getElementById('trainEpochsSelect').value;
      const btn = document.getElementById('btnStartTrain');
      if (btn) btn.textContent = '⏳ Iniciando...';
      try {
        const res = await fetch('/api/train/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ epochs: epochs, batch_size: 4 })
        });
        const data = await res.json();
        if (res.status === 409) {
          alert('⚠️ ' + data.error);
        } else {
          alert('🚀 ' + data.message);
        }
      } catch (e) {
        alert('Error al iniciar entrenamiento: ' + e);
      } finally {
        setTimeout(() => { if (btn) btn.textContent = '🚀 Iniciar Entrenamiento'; }, 1500);
      }
    }

    async function stopSupervisedTraining() {
      try {
        const res = await fetch('/api/train/stop', { method: 'POST' });
        const data = await res.json();
        alert('⏸️ ' + data.message);
      } catch (e) {
        alert('Error al pausar: ' + e);
      }
    }

    function startStatusPolling() {
      setInterval(fetchStatus, 3000);
    }

    window.onload = initApp;
  