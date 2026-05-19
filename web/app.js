(() => {
  const state = {
    manifest: null,
    filter: "needs-review",
    folder: null,
    current: null,
    imageEl: null,
    konva: null,
    rowsByFilename: new Map(),
    autoTone: true,
    // Bumped on every Auto Tone toggle to bust the browser image cache.
    cacheBuster: Date.now(),
  };

  const $ = (s) => document.querySelector(s);

  function toast(msg, ms = 1800) {
    let el = document.getElementById("toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "toast";
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), ms);
  }

  // ---------- View routing ----------
  function showView(name) {
    $("#folder-view").hidden = name !== "folder";
    $("#list-view").hidden = name !== "list";
    $("#editor-view").hidden = name !== "editor";
    $("#topbar").hidden = name === "folder";
    // The review-progress bar belongs to the list view only. renderCounts
    // will re-show it (if there are entries) the next time it runs.
    if (name !== "list") $("#review-progress").hidden = true;
  }

  // ---------- Folder picker ----------
  async function openFolder(path) {
    const errEl = $("#folder-error");
    errEl.hidden = true;
    $("#btn-open-folder").disabled = true;
    $("#folder-progress").hidden = false;
    setProgress({ running: true, total: 0, done: 0, current: "" });
    try {
      const r = await fetch("/api/open-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(j.detail || `HTTP ${r.status}`);
      }
      const j = await r.json();
      state.folder = j.folder;

      // Stay on the folder picker (with the processing bar) until the
      // pipeline finishes; only then switch to the list view.
      while (true) {
        await new Promise((res) => setTimeout(res, 350));
        const pr = await fetch("/api/progress");
        const p = await pr.json();
        setProgress(p);
        if (p.error) throw new Error(p.error);
        if (!p.running) break;
      }

      await fetchManifest();
      $("#folder-info").textContent = state.folder;
      renderCounts();
      renderList();
      showView("list");
    } catch (err) {
      errEl.textContent = String(err.message || err);
      errEl.hidden = false;
    } finally {
      $("#btn-open-folder").disabled = false;
      $("#folder-progress").hidden = true;
    }
  }

  function setProgress(p) {
    const total = p.total || 0;
    const done = p.done || 0;
    const pct = total > 0 ? Math.round((100 * done) / total) : 0;
    $("#folder-progress-fill").style.width = pct + "%";
    let label = total > 0 ? `${done} / ${total} (${pct}%)` : "Discovering files…";
    if (typeof p.eta_seconds === "number" && p.eta_seconds > 0) {
      label += ` · ETA ${formatEta(p.eta_seconds)}`;
    }
    $("#folder-progress-text").textContent = label;
    $("#folder-progress-current").textContent = p.current ? "→ " + p.current : "";
  }

  function formatEta(s) {
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const sec = s % 60;
    if (m < 60) return sec > 0 ? `${m}m ${sec}s` : `${m}m`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
  }

  async function fetchManifest() {
    const r = await fetch("/api/manifest");
    if (!r.ok) throw new Error("manifest fetch failed");
    state.manifest = await r.json();
    state.sizeClusters = computeSizeClusters(state.manifest.entries || []);
  }

  // ---------- Size-cluster prediction ----------
  // Mirrors pipeline.cluster_by_size in Python. Two crops belong to the
  // same cluster if both their long and short edges agree within TOL.
  const SIZE_CLUSTER_TOLERANCE = 0.07;
  const PRINT_TYPES = [
    { label: "4×6", r: 1.5 },
    { label: "5×7", r: 1.4 },
    { label: "8×10", r: 1.25 },
    { label: "3×5", r: 1.667 },
    { label: "3×4", r: 1.333 },
    { label: "polaroid", r: 1.166 },
    { label: "square", r: 1.0 },
    { label: "panoramic", r: 2.0 },
  ];

  function _median(nums) {
    const a = [...nums].sort((x, y) => x - y);
    const mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }

  function computeSizeClusters(entries) {
    const sized = [];
    for (const e of entries) {
      if (!e.crop_box) continue;
      const lo = Math.max(e.crop_box.w, e.crop_box.h);
      const sh = Math.min(e.crop_box.w, e.crop_box.h);
      sized.push({ filename: e.filename, lo, sh });
    }
    sized.sort((a, b) => b.lo - a.lo);
    const clusters = [];
    for (const { filename, lo, sh } of sized) {
      let matched = false;
      for (const c of clusters) {
        if (
          Math.abs(lo - c.median_long) / c.median_long < SIZE_CLUSTER_TOLERANCE &&
          Math.abs(sh - c.median_short) / c.median_short < SIZE_CLUSTER_TOLERANCE
        ) {
          c.members.push(filename);
          c._longs.push(lo);
          c._shorts.push(sh);
          c.median_long = _median(c._longs);
          c.median_short = _median(c._shorts);
          matched = true;
          break;
        }
      }
      if (!matched) {
        clusters.push({
          median_long: lo,
          median_short: sh,
          members: [filename],
          _longs: [lo],
          _shorts: [sh],
        });
      }
    }
    return clusters.map((c) => ({
      median_long: c.median_long,
      median_short: c.median_short,
      members: c.members,
    }));
  }

  function aspectLabel(longEdge, shortEdge) {
    if (!shortEdge) return "";
    const r = longEdge / shortEdge;
    let best = null;
    let bestErr = Infinity;
    for (const t of PRINT_TYPES) {
      const err = Math.abs(r - t.r) / t.r;
      if (err < bestErr) { bestErr = err; best = t; }
    }
    return bestErr < 0.04 ? best.label : "";
  }

  function clusterLabel(c) {
    const lo = Math.round(c.median_long);
    const sh = Math.round(c.median_short);
    const a = aspectLabel(c.median_long, c.median_short);
    const prefix = a ? `${a} · ` : "";
    const n = c.members.length;
    return `${prefix}${lo}×${sh} · ${n} photo${n === 1 ? "" : "s"}`;
  }

  function clusterIndexForFilename(filename) {
    const list = state.sizeClusters || [];
    for (let i = 0; i < list.length; i++) {
      if (list[i].members.includes(filename)) return i;
    }
    return -1;
  }

  function filteredEntries() {
    const entries = state.manifest?.entries ?? [];
    if (state.filter === "all") return entries;
    return entries.filter((e) => e.decision === state.filter);
  }

  function renderCounts() {
    const counts = { approved: 0, "needs-review": 0, rejected: 0, "no-detection": 0 };
    for (const e of state.manifest?.entries ?? []) counts[e.decision] = (counts[e.decision] || 0) + 1;
    $("#counts").textContent =
      `${counts.approved} saved · ${counts["needs-review"]} needs review`;
    // Top-of-list progress bar — fraction of the batch already saved.
    const bar = $("#review-progress");
    const fill = $("#review-progress-fill");
    const label = $("#review-progress-label");
    const total = counts.approved + counts["needs-review"] + counts.rejected + counts["no-detection"];
    if (bar) {
      // While a detection batch is running, the bar shows processing
      // progress instead of save progress (the latter would be near 0
      // and unhelpful early in the run).
      if (state.processing && state.batchProgress && state.batchProgress.total > 0) {
        const bp = state.batchProgress;
        const pct = bp.total > 0 ? Math.round((bp.done / bp.total) * 100) : 0;
        fill.style.width = pct + "%";
        label.textContent = `Processing ${bp.done} / ${bp.total} (${pct}%)`;
        bar.hidden = false;
      } else if (total > 0) {
        const pct = Math.round((counts.approved / total) * 100);
        fill.style.width = pct + "%";
        label.textContent = `${counts.approved} / ${total} (${pct}%)`;
        bar.hidden = false;
      } else {
        bar.hidden = true;
      }
    }
  }

  // ---------- List view ----------
  function renderList() {
    const list = $("#list");
    list.innerHTML = "";
    state.rowsByFilename.clear();
    const entries = filteredEntries();
    if (entries.length === 0) {
      list.innerHTML = `<div style="color:var(--muted);text-align:center;padding:60px;">No photos in this filter.</div>`;
      return;
    }
    for (const e of entries) {
      const row = renderRow(e);
      state.rowsByFilename.set(e.filename, row);
      list.appendChild(row);
    }
    lazyLoad();
  }

  function loadImagesIn(row) {
    row.querySelectorAll("img[data-src]").forEach((img) => {
      img.src = img.dataset.src;
      img.removeAttribute("data-src");
    });
  }

  function refreshRow(entry) {
    // Update in-place: replace the row element and update the manifest cache.
    const idx = state.manifest.entries.findIndex((e) => e.filename === entry.filename);
    if (idx >= 0) state.manifest.entries[idx] = entry;
    const old = state.rowsByFilename.get(entry.filename);
    if (!old) return;
    const fresh = renderRow(entry);
    state.rowsByFilename.set(entry.filename, fresh);
    old.parentNode.replaceChild(fresh, old);
    // Force-load images for the visible row (no need to wait for IntersectionObserver).
    loadImagesIn(fresh);
  }

  function appendNewEntries() {
    // Incremental update used by the in-progress poll loop: just append
    // rows for entries not already in the DOM. Existing rows stay
    // untouched so their images don't reload.
    const list = $("#list");
    const entries = filteredEntries();
    let added = 0;
    for (const e of entries) {
      if (state.rowsByFilename.has(e.filename)) continue;
      const row = renderRow(e);
      state.rowsByFilename.set(e.filename, row);
      list.appendChild(row);
      added++;
    }
    if (added > 0) {
      // Only observe newly-attached imgs (already-loaded ones lose data-src).
      const io = new IntersectionObserver(
        (entries, ob) => {
          for (const ent of entries) {
            if (ent.isIntersecting) {
              const img = ent.target;
              img.src = img.dataset.src;
              img.removeAttribute("data-src");
              ob.unobserve(img);
            }
          }
        },
        { rootMargin: "400px" }
      );
      document.querySelectorAll("#list img[data-src]").forEach((img) => io.observe(img));
    }
  }

  function refreshCroppedImageOnly(entry) {
    // Update the manifest cache and just swap the "auto-cropped" <img>'s
    // src — leaves the "original + green guide" image alone so it doesn't
    // re-fetch when only the cropped output changed (Auto Tone toggle, etc.).
    const idx = state.manifest.entries.findIndex((x) => x.filename === entry.filename);
    if (idx >= 0) state.manifest.entries[idx] = entry;
    const row = state.rowsByFilename.get(entry.filename);
    if (!row) return;
    const img = row.querySelector(".cell.cropped img");
    if (!img) return;
    // Clear any leftover optimistic-spin transform from rotateRow so the
    // new (server-rotated) image doesn't get an extra CSS rotation on top.
    img.style.transition = "";
    img.style.transform = "";
    delete img.dataset.optimisticRot;
    const ts = entry.timestamp || "";
    img.src = `/api/output/${encodeURIComponent(entry.filename)}?t=${encodeURIComponent(ts)}&v=${Date.now()}`;
    img.removeAttribute("data-src");
  }

  function animateRemoveRow(filename) {
    const row = state.rowsByFilename.get(filename);
    if (!row) return;
    // Pin current height so max-height:0 has a starting value to animate FROM.
    row.style.maxHeight = row.scrollHeight + "px";
    // Force a reflow so the pinned value is committed before the class flip.
    void row.offsetHeight;
    row.classList.add("removing");
    state.rowsByFilename.delete(filename);
    setTimeout(() => {
      if (row.parentNode) row.parentNode.removeChild(row);
    }, 450);
  }

  // After a manifest mutation: refresh the row in place if the entry still
  // matches the current filter, otherwise slide it out so the rest of the
  // list moves up to fill the gap.
  function applyEntryUpdate(entry) {
    const idx = state.manifest.entries.findIndex((e) => e.filename === entry.filename);
    if (idx >= 0) state.manifest.entries[idx] = entry;
    if (state.filter === "all" || entry.decision === state.filter) {
      refreshRow(entry);
    } else {
      animateRemoveRow(entry.filename);
    }
  }

  function renderRow(e) {
    const row = document.createElement("div");
    row.className = `list-row ${e.decision}`;
    row.dataset.filename = e.filename;

    // Cropped (left)
    const cropped = document.createElement("div");
    cropped.className = "cell cropped";
    const cropLabel = document.createElement("span");
    cropLabel.className = "cell-label";
    cropLabel.textContent = "Auto-cropped";
    cropped.appendChild(cropLabel);
    if (e.decision === "no-detection") {
      cropped.classList.add("no-output");
      cropped.textContent = "no auto crop";
      cropped.appendChild(cropLabel);
    } else {
      const cimg = document.createElement("img");
      cimg.loading = "lazy";
      cimg.dataset.src = `/api/output/${encodeURIComponent(e.filename)}?t=${e.timestamp || ""}&v=${state.cacheBuster}`;
      cropped.appendChild(cimg);
    }
    cropped.addEventListener("click", () => openEditor(e));

    // Original + guide (right)
    const original = document.createElement("div");
    original.className = "cell original";
    const origLabel = document.createElement("span");
    origLabel.className = "cell-label";
    origLabel.textContent = "Original · detected edges in green";
    original.appendChild(origLabel);
    const oimg = document.createElement("img");
    oimg.loading = "lazy";
    oimg.dataset.src = `/api/original-with-guide/${encodeURIComponent(e.filename)}?t=${e.timestamp || ""}`;
    original.appendChild(oimg);

    // Info + actions (right)
    const info = document.createElement("div");
    info.className = "row-info";
    const autoToneOn = e.auto_tone !== false;
    info.innerHTML = `
      <div class="row-filename">${escapeHtml(e.filename)}</div>
      <div class="row-meta">score: ${e.score ?? 0}</div>
      <span class="badge ${e.decision}">${e.decision.replace("-", " ")}</span>
      <label class="row-toggle" title="Per-photo Auto Tone (per-channel histogram stretch). On by default.">
        <input type="checkbox" data-action="auto-tone" ${autoToneOn ? "checked" : ""} />
        <span>Auto Tone</span>
      </label>
      <div class="row-actions">
        <button class="approve" data-action="approve">Save</button>
        <button data-action="edit">Edit</button>
        <div class="rot-row rotate-90">
          <button class="icon-btn" data-action="rot-ccw" title="Rotate 90° counter-clockwise" aria-label="Rotate 90° counter-clockwise">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M3 12a9 9 0 1 0 3-6.7"/>
              <polyline points="3 3 3 9 9 9"/>
            </svg>
          </button>
          <button class="icon-btn" data-action="rot-cw" title="Rotate 90° clockwise" aria-label="Rotate 90° clockwise">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M21 12a9 9 0 1 1-3-6.7"/>
              <polyline points="21 3 21 9 15 9"/>
            </svg>
          </button>
        </div>
      </div>
    `;
    const autoToneCb = info.querySelector("input[data-action='auto-tone']");
    if (autoToneCb) {
      autoToneCb.addEventListener("change", () => toggleEntryAutoTone(e.filename, autoToneCb.checked));
    }
    info.addEventListener("click", (ev) => {
      const a = ev.target.closest("[data-action]");
      if (!a) return;
      const action = a.dataset.action;
      if (action === "approve") setDecision(e.filename, "approved");
      else if (action === "edit") openEditor(e);
      else if (action === "rot-cw") rotateRow(e.filename, 1);
      else if (action === "rot-ccw") rotateRow(e.filename, -1);
    });

    row.appendChild(cropped);
    row.appendChild(original);
    row.appendChild(info);
    return row;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function lazyLoad() {
    const io = new IntersectionObserver(
      (entries, ob) => {
        for (const ent of entries) {
          if (ent.isIntersecting) {
            const img = ent.target;
            img.src = img.dataset.src;
            ob.unobserve(img);
          }
        }
      },
      { rootMargin: "400px" }
    );
    document.querySelectorAll("#list img[data-src]").forEach((img) => io.observe(img));
  }

  async function setDecision(filename, decision) {
    const row = state.rowsByFilename.get(filename);
    if (row) row.classList.add("updating");
    const r = await fetch(`/api/decision/${encodeURIComponent(filename)}?decision=${decision}`, { method: "POST" });
    if (!r.ok) {
      if (row) row.classList.remove("updating");
      toast("Decision update failed");
      return;
    }
    // Decision endpoint doesn't return the new entry; fetch the manifest and
    // pluck just this one out.
    await fetchManifest();
    const entry = state.manifest.entries.find((e) => e.filename === filename);
    renderCounts();
    if (entry) applyEntryUpdate(entry);
    toast(decision === "approved" ? `Saved ${filename}` : `${filename} marked ${decision}`);
  }

  async function snapRowToSize(filename, longEdge, shortEdge) {
    const row = state.rowsByFilename.get(filename);
    if (row) row.classList.add("updating");
    try {
      const r = await fetch("/api/snap-to-size", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, long_edge: longEdge, short_edge: shortEdge }),
      });
      if (!r.ok) {
        toast("Snap to size failed");
        if (row) row.classList.remove("updating");
        return;
      }
      const j = await r.json();
      refreshRow(j.entry);
      // Recompute clusters AFTER the manifest cache is updated, so the
      // snapped photo's new size feeds back into the cluster medians.
      state.sizeClusters = computeSizeClusters(state.manifest.entries || []);
    } catch (err) {
      console.error(err);
      toast("Snap error");
      if (row) row.classList.remove("updating");
    }
  }

  async function rotateRow(filename, delta) {
    const row = state.rowsByFilename.get(filename);
    const img = row?.querySelector(".cell.cropped img");
    // Optimistic spin: rotate the existing thumbnail in the browser
    // right away so the click feels instant. The server's re-rendered
    // JPEG arrives a moment later and seamlessly replaces it.
    //
    // Note: we DON'T mod the angle to [0, 360). CSS `transform: rotate`
    // animates between the literal previous and next values, so wrapping
    // through 0° would make the image spin the long way round (e.g.
    // 0° → 270° goes 3/4 turn clockwise instead of 1/4 turn left).
    if (img) {
      const prev = parseFloat(img.dataset.optimisticRot || "0") || 0;
      const next = prev + delta * 90;
      img.dataset.optimisticRot = String(next);
      img.style.transition = "transform 0.18s ease";
      img.style.transform = `rotate(${next}deg)`;
    }
    try {
      const r = await fetch("/api/rotate-upright", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, delta }),
      });
      if (!r.ok) {
        toast("Rotate failed");
        // Roll back the visual
        if (img) {
          img.style.transition = "";
          img.style.transform = "";
          delete img.dataset.optimisticRot;
        }
        return;
      }
      const j = await r.json();
      // Replace just the cropped <img> src. Once the new image loads,
      // clear the CSS rotation so the actual rotated bytes show without
      // a double-rotation.
      if (img) {
        const swap = new Image();
        swap.onload = () => {
          img.style.transition = "";
          img.style.transform = "";
          delete img.dataset.optimisticRot;
          img.src = swap.src;
        };
        const ts = j.entry.timestamp || "";
        swap.src = `/api/output/${encodeURIComponent(filename)}?t=${encodeURIComponent(ts)}&v=${Date.now()}`;
      }
      // Update the manifest cache so subsequent actions see the latest entry.
      const idx = state.manifest.entries.findIndex((x) => x.filename === filename);
      if (idx >= 0) state.manifest.entries[idx] = j.entry;
    } catch (err) {
      console.error(err);
      toast("Rotate error");
      if (img) {
        img.style.transition = "";
        img.style.transform = "";
        delete img.dataset.optimisticRot;
      }
    }
  }

  // ---------- Editor (unchanged structure, reference panel removed) ----------
  function openEditor(entry) {
    if (!entry || !entry.filename) return;
    // Remember where the user was in the list. Hiding the list with
    // display:none collapses the document height, which makes the browser
    // clamp the scroll position to 0; we restore this value on close.
    state.savedListScroll = window.scrollY || document.documentElement.scrollTop || 0;
    state.current = entry;
    // Session-scoped upright rotation: starts at the manifest value, the
    // ↺/↻ buttons nudge it, and "Reset to auto" restores it. Saved only
    // when the user clicks Approve/Reject.
    state.uprightQt = ((entry.upright_rotation_qt ?? 0) % 4 + 4) % 4;
    $("#editor-filename").textContent = entry.filename;
    showView("editor");

    const img = new Image();
    img.onload = () => {
      state.imageEl = img;
      requestAnimationFrame(() => requestAnimationFrame(setupKonva));
    };
    img.onerror = () => {
      toast(`Image failed to load: ${entry.filename}`, 4000);
    };
    img.src = `/api/image/${encodeURIComponent(entry.filename)}`;
  }

  function editorHasUnsavedEdits() {
    if (!state.current || !state.konva) return false;
    const slider = parseFloat($("#rotation-slider").value);
    const savedRot = state.current.rotation_deg ?? 0;
    if (Math.abs(slider - savedRot) > 0.01) return true;
    const savedUp = (((state.current.upright_rotation_qt ?? 0) % 4) + 4) % 4;
    const curUp = (((state.uprightQt || 0) % 4) + 4) % 4;
    if (savedUp !== curUp) return true;
    const crop = getCropBoxInImageCoords();
    const saved = state.current.crop_box;
    if (!crop || !saved) return !!crop !== !!saved;
    if (Math.abs(crop.cx - saved.cx) > 0.5) return true;
    if (Math.abs(crop.cy - saved.cy) > 0.5) return true;
    if (Math.abs(crop.w - saved.w) > 0.5) return true;
    if (Math.abs(crop.h - saved.h) > 0.5) return true;
    return false;
  }

  async function closeEditorWithSave() {
    if (editorHasUnsavedEdits()) {
      // Persist the user's tweaks but keep the entry's existing decision so
      // hitting Back doesn't promote a needs-review item to approved.
      await saveDecision(state.current.decision || "needs-review");
      return;
    }
    closeEditor();
  }

  function closeEditor() {
    if (state.konva) {
      state.konva.stage.destroy();
      state.konva = null;
    }
    state.imageEl = null;
    const justEditedFilename = state.current?.filename;
    state.current = null;
    showView("list");
    // Restore scroll on the next frame so the list is laid out first.
    const y = state.savedListScroll ?? 0;
    requestAnimationFrame(() => {
      window.scrollTo({ top: y, left: 0, behavior: "auto" });
      // Make sure the row the user just edited is visible — handy when the
      // list was re-rendered (e.g. after an approve / reject changed which
      // filter tab the row belongs to).
      if (justEditedFilename) {
        const row = state.rowsByFilename.get(justEditedFilename);
        if (row && !isInViewport(row)) {
          row.scrollIntoView({ block: "center", behavior: "auto" });
        }
      }
    });
  }

  function isInViewport(el) {
    const r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < (window.innerHeight || document.documentElement.clientHeight);
  }

  function setupKonva() {
    if (typeof Konva === "undefined") {
      toast("Konva failed to load");
      return;
    }
    const container = $("#canvas-container");
    container.innerHTML = "";
    const W = container.clientWidth || container.parentElement.clientWidth;
    const H = container.clientHeight || container.parentElement.clientHeight;
    if (W < 10 || H < 10) {
      requestAnimationFrame(setupKonva);
      return;
    }
    const img = state.imageEl;
    if (!img) return;
    const fit = Math.min(W / img.width, H / img.height);
    const displayW = img.width * fit;
    const displayH = img.height * fit;

    const stage = new Konva.Stage({ container, width: W, height: H });
    const imageLayer = new Konva.Layer();
    const overlayLayer = new Konva.Layer();
    stage.add(imageLayer);
    stage.add(overlayLayer);

    // Both the image and overlays live inside ONE rotating group. That
    // guarantees the crop rectangle, the green detected-edge guide, and
    // the image content all transform together — they can never drift out
    // of alignment regardless of rotation sign conventions.
    //
    // Konva positive = visual CW, cv2 positive = visual CCW. Negate at the
    // boundary so the photo appears upright after rotation.
    // upright_rotation_qt adds k*90° CW (post-crop quarter-turns) to the
    // preview so it matches what the saved file will look like.
    const konvaRot = (v) => -(v ?? 0);
    const uprightDeg = () => (state.uprightQt || 0) * 90;
    const group = new Konva.Group({
      x: W / 2, y: H / 2,
      offsetX: displayW / 2, offsetY: displayH / 2,
      rotation: konvaRot(state.current.rotation_deg) + uprightDeg(),
    });
    const konvaImg = new Konva.Image({ image: img, x: 0, y: 0, width: displayW, height: displayH });
    group.add(konvaImg);

    // Crop rect lives INSIDE the group, expressed in image-pixel coords
    // (scaled by `fit`). Its own rotation cancels the group's rotation, so
    // it renders axis-aligned in canvas while still tracking the photo as
    // the rotation slider moves. Use the detected corners' centroid as
    // center when available — that guarantees overlap with the green guide.
    // The blue crop rect always reflects the manifest's saved crop_box —
    // that's the rect that will actually be saved to disk, after any
    // shrink (CROP_SHRINK in straighten.py) and any user edits. The green
    // guide uses detected_corners separately, so it shows the full
    // detected edge for visual reference.
    const corners = state.current.detected_corners;
    const cb = state.current.crop_box;
    let cx, cy, w, h;
    if (cb) {
      cx = cb.cx; cy = cb.cy; w = cb.w; h = cb.h;
    } else if (corners && corners.length === 4) {
      cx = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4;
      cy = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4;
      const dx01 = corners[1].x - corners[0].x;
      const dy01 = corners[1].y - corners[0].y;
      const dx12 = corners[2].x - corners[1].x;
      const dy12 = corners[2].y - corners[1].y;
      const side01 = Math.hypot(dx01, dy01);
      const side12 = Math.hypot(dx12, dy12);
      w = Math.max(side01, side12);
      h = Math.min(side01, side12);
    } else {
      cx = img.width / 2; cy = img.height / 2;
      w = img.width * 0.8; h = img.height * 0.8;
    }
    const cropRect = new Konva.Rect({
      x: cx * fit, y: cy * fit,
      width: w * fit, height: h * fit,
      offsetX: (w * fit) / 2, offsetY: (h * fit) / 2,
      // Cancel the group's rotation so the rect is axis-aligned in canvas.
      rotation: -konvaRot(state.current.rotation_deg),
      stroke: "#4f8cff", strokeWidth: 2,
      fill: "rgba(0,0,0,0.001)",
      draggable: true,
    });
    group.add(cropRect);

    // Green detected-edge guide, in image-pixel coords inside the same group.
    if (corners && corners.length === 4) {
      const points = [];
      for (const pt of corners) points.push(pt.x * fit, pt.y * fit);
      const guide = new Konva.Line({
        points, closed: true,
        stroke: "#2ecc71", strokeWidth: 3,
        dash: [12, 8], listening: false,
        shadowColor: "#000", shadowBlur: 2, shadowOpacity: 0.6,
      });
      group.add(guide);
    }
    imageLayer.add(group);
    imageLayer.draw();

    const transformer = new Konva.Transformer({
      nodes: [cropRect],
      rotateEnabled: false, keepRatio: false,
      borderStroke: "#4f8cff", anchorStroke: "#4f8cff", anchorFill: "#0f1115",
      anchorSize: 12,
      enabledAnchors: ["top-left", "top-right", "bottom-left", "bottom-right",
        "top-center", "bottom-center", "middle-left", "middle-right"],
      boundBoxFunc: (oldBox, newBox) => {
        if (newBox.width < 20 || newBox.height < 20) return oldBox;
        return newBox;
      },
    });

    // Hit zones, edges-take-priority model:
    //   - A generous strip near each edge (extending both outward AND
    //     inward) catches edge resize. Corners get the four diagonal
    //     regions. The center of the rect — anywhere more than INWARD
    //     from every edge — is the drag-to-move zone.
    //   - The inward depth (~1/4 of the rect's smallest side, capped at
    //     80px) scales with rect size so small crops aren't entirely
    //     resize-zones.
    const HUGE = 4000;
    transformer.anchorStyleFunc((anchor) => {
      const name = anchor.name();
      anchor.hitFunc((ctx, shape) => {
        const r = cropRect;
        const rw = r.width() * r.scaleX();
        const rh = r.height() * r.scaleY();
        // How deep into the rect each edge's grab zone reaches.
        const inward = Math.max(20, Math.min(80, Math.min(rw, rh) * 0.25));
        // Per-anchor diagonal reserve at the rect ends so corners aren't
        // swallowed by adjacent edges.
        const corner = Math.max(24, Math.min(60, Math.min(rw, rh) * 0.2));
        let x, y, w, h;
        if (name === "middle-left") {
          x = -HUGE; y = -rh / 2 + corner;
          w = HUGE + inward; h = rh - 2 * corner;
        } else if (name === "middle-right") {
          x = -inward; y = -rh / 2 + corner;
          w = HUGE + inward; h = rh - 2 * corner;
        } else if (name === "top-center") {
          x = -rw / 2 + corner; y = -HUGE;
          w = rw - 2 * corner; h = HUGE + inward;
        } else if (name === "bottom-center") {
          x = -rw / 2 + corner; y = -inward;
          w = rw - 2 * corner; h = HUGE + inward;
        } else {
          // Corner zones — diagonally outside, with a small inward overlap
          // so the corner itself is grabbable inside the rect too.
          const isLeft = name.includes("left");
          const isTop = name.includes("top");
          x = isLeft ? -HUGE : -inward;
          y = isTop ? -HUGE : -inward;
          w = HUGE + inward; h = HUGE + inward;
        }
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.closePath();
        ctx.fillStrokeShape(shape);
      });
    });

    overlayLayer.add(transformer);
    overlayLayer.draw();

    state.konva = {
      stage, imageLayer, overlayLayer, group,
      konvaImg, cropRect, transformer,
      displayW, displayH, fit, W, H,
    };
    $("#rotation-slider").value = String(state.current.rotation_deg ?? 0);
    updateRotationReadout();
  }

  function updateRotationReadout() {
    const v = parseFloat($("#rotation-slider").value);
    $("#rotation-readout").textContent = `${v.toFixed(1)}°`;
    if (state.konva) {
      // Slider value is in cv2/manifest convention; negate for Konva.
      // The upright quarter-turn offset is applied to the group only —
      // the rect cancels just the small straighten rotation so it tracks
      // the photo through 90° turns while still being axis-aligned w.r.t.
      // the photo's image axes.
      const kv = -v;
      const upright = (state.uprightQt || 0) * 90;
      state.konva.group.rotation(kv + upright);
      state.konva.cropRect.rotation(-kv);
      // Transformer doesn't auto-track parent-group transform changes;
      // force it so handles stay glued to the rect after a 90° turn.
      if (state.konva.transformer) state.konva.transformer.forceUpdate();
      state.konva.imageLayer.batchDraw();
      state.konva.overlayLayer.batchDraw();
    }
  }

  function rotateUprightBy(delta) {
    if (!state.current) return;
    state.uprightQt = (((state.uprightQt || 0) + delta) % 4 + 4) % 4;
    updateRotationReadout();
  }

  function getCropBoxInImageCoords() {
    const k = state.konva;
    if (!k) return null;
    const r = k.cropRect;
    // Read the effective rect WITHOUT mutating its internal state. Earlier
    // versions reset scaleX/scaleY here, but that produced a visible jump
    // on Save because the rect's offsetX/offsetY (set in setupKonva) didn't
    // update along with width/height, so the centre shifted by a few pixels.
    const sx = r.scaleX(), sy = r.scaleY();
    const w = r.width() * sx, h = r.height() * sy;
    const local_x = r.x();
    const local_y = r.y();
    return { cx: local_x / k.fit, cy: local_y / k.fit, w: w / k.fit, h: h / k.fit };
  }

  async function saveDecision(decision) {
    if (!state.current) return;
    const rotation = parseFloat($("#rotation-slider").value);
    const crop = getCropBoxInImageCoords();
    if (!crop) return;
    // Visual feedback during the save round-trip (server re-renders the
    // full-res crop + writes JPEG + updates the manifest — easily 1–2s).
    const btn = $("#btn-approve");
    const originalLabel = btn.textContent;
    btn.disabled = true;
    btn.classList.add("saving");
    btn.innerHTML = '<span class="btn-spinner"></span> Saving…';
    const restore = () => {
      btn.disabled = false;
      btn.classList.remove("saving");
      btn.textContent = originalLabel;
    };
    try {
      const r = await fetch("/api/save-edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: state.current.filename,
          rotation_deg: rotation,
          crop_box: crop,
          decision,
          upright_rotation_qt: state.uprightQt || 0,
        }),
      });
      if (!r.ok) {
        toast("Save failed");
        return;
      }
      toast("Saved");
      // Bust browser image caches so the row reloads the freshly-written
      // crop instead of any leftover toned/untoned URL from before.
      state.cacheBuster = Date.now();
      const savedFilename = state.current.filename;
      await fetchManifest();
      renderCounts();
      closeEditor();
      // Apply the targeted update (animate-out if it no longer matches the
      // active filter; otherwise refresh in place). Done after closeEditor
      // so the user sees the list with the saved row sliding away.
      const entry = state.manifest.entries.find((e) => e.filename === savedFilename);
      if (entry) applyEntryUpdate(entry);
    } catch (err) {
      console.error(err);
      toast("Save failed");
    } finally {
      // Always restore the button — otherwise it stays stuck in "Saving…"
      // for the next photo the user opens.
      restore();
    }
  }

  // ---------- Re-run detection from the editor ----------
  // The list view has a size-type dropdown that drives /api/snap-to-size;
  // the editor uses the same endpoint but with the entry's current crop
  // dimensions as the aspect target.
  async function redetectInEditorWithAspect() {
    if (!state.current || !state.current.crop_box) return;
    const lo = Math.max(state.current.crop_box.w, state.current.crop_box.h);
    const sh = Math.min(state.current.crop_box.w, state.current.crop_box.h);
    const btn = $("#btn-detect-another");
    btn.disabled = true;
    try {
      const r = await fetch("/api/snap-to-size", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: state.current.filename,
          long_edge: lo,
          short_edge: sh,
        }),
      });
      if (!r.ok) { toast("Redetect failed"); return; }
      const j = await r.json();
      state.current = j.entry;
      setupKonva();
      await fetchManifest();
    } finally {
      btn.disabled = false;
    }
  }

  // ---------- Size clustering ----------
  let _clusterCache = [];

  async function openClusterModal() {
    const r = await fetch("/api/size-clusters");
    if (!r.ok) {
      toast("Failed to fetch size groups");
      return;
    }
    const j = await r.json();
    _clusterCache = j.clusters || [];
    const body = $("#cluster-body");
    body.innerHTML = "";
    if (_clusterCache.length === 0) {
      body.innerHTML = `<p style="color:var(--muted);">No size groups detected. Run detection on a folder first.</p>`;
    } else {
      _clusterCache.forEach((c, i) => {
        const row = document.createElement("div");
        row.className = "cluster-row";
        const dimLong = Math.round(c.median_long);
        const dimShort = Math.round(c.median_short);
        row.innerHTML = `
          <input type="checkbox" data-idx="${i}" ${c.members.length > 1 ? "checked" : ""} />
          <div class="ci-dim">${dimLong} × ${dimShort}</div>
          <div class="ci-count">${c.members.length} photo${c.members.length === 1 ? "" : "s"}</div>
          <div class="ci-members">${escapeHtml(c.members.join(", "))}</div>
        `;
        body.appendChild(row);
      });
    }
    $("#cluster-modal").hidden = false;
  }

  async function applyClusters() {
    const checks = document.querySelectorAll("#cluster-body input[type=checkbox]");
    const selected = [];
    checks.forEach((c) => {
      if (c.checked) {
        const idx = parseInt(c.dataset.idx, 10);
        if (!isNaN(idx) && _clusterCache[idx]) selected.push(_clusterCache[idx]);
      }
    });
    if (selected.length === 0) {
      $("#cluster-modal").hidden = true;
      return;
    }
    $("#cluster-apply").disabled = true;
    $("#cluster-apply").textContent = "Applying…";
    try {
      const r = await fetch("/api/apply-size-clusters", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clusters: selected }),
      });
      if (!r.ok) {
        toast("Apply failed");
      } else {
        const j = await r.json();
        toast(`Updated ${j.updated.length} photo${j.updated.length === 1 ? "" : "s"}`);
        await fetchManifest();
        renderCounts();
        renderList();
      }
    } finally {
      $("#cluster-apply").disabled = false;
      $("#cluster-apply").textContent = "Apply selected groups";
      $("#cluster-modal").hidden = true;
    }
  }

  function nudgeRotation(d) { setRotation(parseFloat($("#rotation-slider").value) + d); }
  function setRotation(v) {
    v = Math.max(-45, Math.min(45, v));
    $("#rotation-slider").value = String(v);
    updateRotationReadout();
  }

  // ---------- Bind ----------
  function bind() {
    $("#btn-open-folder").addEventListener("click", () => {
      const v = $("#folder-path").value.trim();
      if (v) openFolder(v);
    });
    $("#folder-path").addEventListener("keydown", (e) => {
      if (e.key === "Enter") $("#btn-open-folder").click();
    });

    const gotoFolderPicker = () => {
      $("#folder-path").value = state.folder || "";
      showView("folder");
    };
    $("#btn-change-folder").addEventListener("click", gotoFolderPicker);
    // Logo / tool-name in the topbar doubles as "go back to the picker".
    document.querySelectorAll("#topbar .title").forEach((el) => {
      el.addEventListener("click", gotoFolderPicker);
    });

    $("#tabs").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-filter]");
      if (!btn) return;
      document.querySelectorAll("#tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.dataset.filter;
      renderList();
    });

    $("#btn-save-all").addEventListener("click", saveAll);

    $("#btn-reprocess").addEventListener("click", async () => {
      const btn = $("#btn-reprocess");
      btn.disabled = true;
      btn.textContent = "Checking…";
      try {
        const r = await fetch("/api/reprocess", { method: "POST" });
        if (!r.ok) {
          toast("Reprocess failed");
          return;
        }
        const j = await r.json();
        if (!j.processed) {
          toast("No new files in this folder");
          return;
        }
        toast(`Added ${j.processed} new file${j.processed === 1 ? "" : "s"}`);
        await fetchManifest();
        renderCounts();
        renderList();
      } finally {
        btn.disabled = false;
        btn.textContent = "Reprocess";
      }
    });

    $("#btn-back").addEventListener("click", closeEditorWithSave);
    $("#rotation-slider").addEventListener("input", updateRotationReadout);
    $("#btn-rotate-left").addEventListener("click", () => nudgeRotation(-0.5));
    $("#btn-rotate-right").addEventListener("click", () => nudgeRotation(0.5));
    $("#btn-upright-ccw").addEventListener("click", () => rotateUprightBy(-1));
    $("#btn-upright-cw").addEventListener("click", () => rotateUprightBy(1));
    $("#btn-reset").addEventListener("click", () => {
      if (!state.current) return;
      state.uprightQt =
        ((state.current.upright_rotation_qt ?? 0) % 4 + 4) % 4;
      setupKonva();
    });
    $("#btn-fit-frame").addEventListener("click", () => {
      if (!state.konva) return;
      const k = state.konva;
      // The crop rect lives inside the rotating image group, with its
      // own offset = size/2 so its (x, y) is its centre. Snap it to the
      // whole image in image-pixel-display coords.
      k.cropRect.width(k.displayW);
      k.cropRect.height(k.displayH);
      k.cropRect.offsetX(k.displayW / 2);
      k.cropRect.offsetY(k.displayH / 2);
      k.cropRect.x(k.displayW / 2);
      k.cropRect.y(k.displayH / 2);
      k.cropRect.scaleX(1);
      k.cropRect.scaleY(1);
      k.imageLayer.batchDraw();
      k.overlayLayer.batchDraw();
    });
    $("#btn-approve").addEventListener("click", () => saveDecision("approved"));
    $("#btn-reset-rotation").addEventListener("click", () => setRotation(0));

    document.addEventListener("keydown", (e) => {
      if ($("#editor-view").hidden) return;
      if (e.target.tagName === "INPUT") return;
      if (e.key === "Escape") closeEditorWithSave();
      else if (e.key === "s" || e.key === "S") saveDecision("approved");
      else if (e.key === "Enter") saveDecision("approved");
      else if (e.key === "ArrowLeft") { nudgeRotation(e.shiftKey ? -0.1 : -0.5); e.preventDefault(); }
      else if (e.key === "ArrowRight") { nudgeRotation(e.shiftKey ? 0.1 : 0.5); e.preventDefault(); }
    });

    window.addEventListener("resize", () => {
      if (!$("#editor-view").hidden && state.imageEl) setupKonva();
    });
  }

  // ---------- Save all (bulk-approve needs-review) ----------
  async function saveAll() {
    const candidates = (state.manifest?.entries ?? []).filter(
      (e) => e.decision === "needs-review"
    );
    if (candidates.length === 0) {
      toast("Nothing in Needs review");
      return;
    }
    if (
      !confirm(
        `Save all ${candidates.length} needs-review photo${candidates.length === 1 ? "" : "s"} as approved?`
      )
    ) return;
    const btn = $("#btn-save-all");
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Saving…";
    try {
      const r = await fetch("/api/save-all", { method: "POST" });
      if (!r.ok) { toast("Save all failed"); return; }
      const j = await r.json();
      toast(`Saved ${j.saved} photo${j.saved === 1 ? "" : "s"}`);
      await fetchManifest();
      renderCounts();
      // If we're looking at the Needs review tab, the rows we just saved
      // no longer belong there — slide them all out in a staggered cascade.
      if (state.filter === "needs-review") {
        const rows = (j.filenames || [])
          .map((fn) => state.rowsByFilename.get(fn))
          .filter(Boolean);
        rows.forEach((row, i) => {
          row.style.maxHeight = row.scrollHeight + "px";
          row.style.transitionDelay = `${i * 30}ms`;
        });
        void $("#list").offsetHeight;
        rows.forEach((row) => row.classList.add("removing"));
        const total = 450 + rows.length * 30;
        setTimeout(() => {
          // Snap to the post-animation state: clear lingering inline styles
          // and re-render so anything that drifted in the manifest is in sync.
          renderList();
        }, total);
      } else {
        renderList();
      }
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  }

  // ---------- Per-photo Auto Tone toggle ----------
  async function toggleEntryAutoTone(filename, on) {
    const row = state.rowsByFilename.get(filename);
    if (row) row.classList.add("updating");
    try {
      const r = await fetch(
        `/api/auto-tone/${encodeURIComponent(filename)}?on=${on}`,
        { method: "POST" }
      );
      if (!r.ok) {
        toast("Auto Tone toggle failed");
        return;
      }
      const j = await r.json();
      // Only the cropped output changed — leave the original-with-guide
      // image alone so the browser doesn't re-fetch it.
      refreshCroppedImageOnly(j.entry);
    } catch (err) {
      console.error(err);
      toast("Auto Tone toggle error");
    } finally {
      if (row) row.classList.remove("updating");
    }
  }

  // ---------- Boot ----------
  (async function init() {
    bind();
    // If the server already has a folder (started via CLI with an arg),
    // skip the picker and jump straight to the list.
    const stateRes = await fetch("/api/state");
    const st = await stateRes.json();
    if (st.version) {
      const v = $("#app-version");
      if (v) v.textContent = `v${st.version}`;
    }
    if (st.folder) {
      state.folder = st.folder;
      try {
        await fetchManifest();
        $("#folder-info").textContent = state.folder;
        renderCounts();
        renderList();
        showView("list");
        return;
      } catch (e) {
        // fall through to folder picker
      }
    }
    showView("folder");
  })();
})();
