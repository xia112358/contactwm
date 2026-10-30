import { ChevronsLeft, ChevronsRight, Focus, FolderOpen, Pause, Play, StepBack, StepForward, createIcons } from 'lucide';

const topbar = () => `
  <header class="topbar">
    <div class="brand"><div><strong>Motion Review</strong><span id="motionLabel">No motion</span></div></div>
    <div class="top-actions">
      <button class="icon" id="openMotionBtn" title="打开轨迹／回放／报告" aria-label="打开轨迹／回放／报告"><i data-lucide="folder-open"></i></button>
      <button class="icon" id="focusBtn" title="Frame scene"><i data-lucide="focus"></i></button>
    </div>
  </header>`;

const viewport = () => `
  <section class="viewport">
    <canvas id="scene"></canvas>
    <div class="viewport-hud"><strong id="viewportMotion">No motion loaded</strong><span id="viewportFrame">Frame 0</span><span id="viewportTime">0.00 s</span></div>
    <div class="shortcut-hint"><span>Space</span> Play <span>← →</span> Frame <span><span>[ ]</span> Cut</span></div>
    <div id="sceneStatus" class="scene-status">Select a motion asset</div>
    <button id="retryLoad" class="retry-load" hidden>Retry loading</button>
  </section>`;

const contactPanel = () => `
  <div class="tab-page active" data-page="contact">
    <div class="page-heading"><strong id="contactHeading">Playback</strong></div>
    <section id="collectionPanel" hidden>
      <label class="stack-field"><span>案例</span><select id="reviewCase" aria-label="案例"></select></label>
      <label class="stack-field"><span>参考轨迹</span><select id="reviewReference" aria-label="参考轨迹"></select></label>
      <label class="stack-field"><span>对比方式</span><select id="reviewLayout" aria-label="对比方式"><option value="beside">并排：参考在 −Y，当前在 +Y</option><option value="overlay">原坐标叠加：蓝色为参考</option></select></label>
      <p id="comparisonClock"></p>
      <label class="stack-field"><span>片段</span><select id="reviewSegment" aria-label="片段"></select></label>
      <p id="reviewMetrics"></p><button id="reviewWorst">定位最大穿透帧</button>
    </section>
    <section id="reviewPanel" hidden></section>
    <p id="playbackNotice" class="playback-notice" hidden>只读轨迹：接触状态以已验证的 Newton 数据为准。</p>
  </div>`;

const displayPanel = () => `
  <div class="tab-page" data-page="display">
    <div class="page-heading"><strong>Display</strong></div>
    <section class="display-section"><div class="section-heading">Scene</div><div class="layer-list">
      <label class="toggle"><input id="terrainToggle" type="checkbox" checked /><span>Terrain mesh</span></label><label class="toggle"><input id="surfaceToggle" type="checkbox" /><span>Contact surfaces</span></label><label class="toggle"><input id="contactSampleToggle" type="checkbox" checked /><span>Contact point + force</span></label><label class="toggle"><input id="rootPathToggle" type="checkbox" /><span>Root path</span></label>
    </div></section>
    <section class="display-section"><div class="section-heading">Camera / Terrain</div>
      <button id="focusRobot">聚焦机器人</button>
      <label class="toggle"><input id="followRobot" type="checkbox" /><span>跟随当前机器人</span></label>
      <label class="slider-field"><span>地形透明度</span><input id="terrainOpacity" aria-label="地形透明度" type="range" min="0.05" max="1" step="0.05" value="1" /></label>
    </section>
    <details id="displayOptions" class="compact-options"><summary>显示细节</summary>
      <label class="slider-field"><span>Force scale</span><input id="forceScale" type="range" min="25" max="300" value="100" /></label>
      <div id="partLegend" class="part-legend"></div>
    </details>
  </div>`;

const inspector = () => `
  <aside class="inspector">
    <div class="inspector-tabs"><button data-tab="contact" class="active">Contact</button><button data-tab="display">Display</button></div>
    <div class="inspector-pages">${contactPanel()}${displayPanel()}</div>
    <details class="recent-motion-dock" aria-label="Motions opened this session"><summary class="recent-motion-head"><span>最近打开</span><small id="recentMotionCount">0</small></summary><div id="recentMotionList" class="recent-motion-list"></div></details>
  </aside>`;

const timeline = () => `
  <section class="timeline-panel">
    <div id="timelineResizer" class="timeline-resizer" title="Drag to resize timeline"><span></span></div>
    <div class="transport"><div class="toolbar-group cut-nav" role="group" aria-label="切分点导航"><button class="icon" id="prevCut" title="上一切分点（到开头后循环）" aria-label="上一切分点"><i data-lucide="chevrons-left"></i></button><span>切分点</span><button class="icon" id="nextCut" title="下一切分点（到末尾后循环）" aria-label="下一切分点"><i data-lucide="chevrons-right"></i></button></div><div class="transport-main"><div class="toolbar-group playback-nav"><button class="icon" id="prevBtn" title="Previous frame"><i data-lucide="step-back"></i></button><button class="icon play" id="playBtn" title="Play or pause"><i data-lucide="play"></i></button><button class="icon" id="nextBtn" title="Next frame"><i data-lucide="step-forward"></i></button></div><div class="frame-readout"><input id="frameInput" class="frame-input" type="number" min="0" value="0" /><span id="frameTotal">/ 0</span></div></div><label class="fps">FPS <input id="fpsInput" type="number" min="1" max="240" value="50" /></label></div>
    <div class="timeline-wrap"><canvas id="timeline" aria-label="Contact timeline"></canvas><div id="timelineHover" class="timeline-hover hidden"></div></div>
  </section>`;

export function mountAppShell(root: HTMLElement): void {
  const template = document.createElement('template');
  template.innerHTML = `${topbar()}<main class="workspace">${viewport()}${inspector()}${timeline()}</main>
    <dialog id="motionPicker" aria-labelledby="motionPickerTitle"><div class="motion-picker-head"><strong id="motionPickerTitle">打开轨迹／回放／报告</strong><button id="closeMotionPicker" aria-label="Close motion picker">关闭</button></div>
      <form id="reviewOpenForm" class="review-open-form"><label for="reviewPath">文件路径</label><input id="reviewPath" required aria-label="回放或案例清单路径" placeholder="NPZ / 报告 / manifest.json / cases.json" /><input id="reviewTerrain" aria-label="地形路径" placeholder="地形 OBJ 路径（可选）" /><button id="reviewOpen" type="submit">打开文件</button></form>
      <label for="motionSearch">已注册轨迹 · 搜索名称或 ID</label><input id="motionSearch" type="search" autocomplete="off" placeholder="搜索轨迹" /><div id="motionPickerList"></div>
    </dialog>`;
  root.replaceChildren(template.content.cloneNode(true));
  refreshIcons();
}

export function refreshIcons(): void {
  createIcons({ icons: { Play, Pause, StepBack, StepForward, Focus, FolderOpen, ChevronsLeft, ChevronsRight } });
}
