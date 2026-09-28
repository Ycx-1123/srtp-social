import test from "node:test";
import assert from "node:assert/strict";
import { LiveController } from "../web/js/live.js";

test("an expired live session stops capture and reports the lost session", async () => {
  const previousFetch = globalThis.fetch;
  const previousWindow = globalThis.window;
  const errors = [];
  let captureStopped = false;
  let clearedTimer = null;
  let renderedState = null;
  const faceStatus = { textContent: "", dataset: {} };
  const speechState = { textContent: "", dataset: {} };

  globalThis.window = {
    setTimeout,
    clearTimeout,
    clearInterval(timer) { clearedTimer = timer; },
  };
  globalThis.fetch = async () => new Response(JSON.stringify({
    detail: {
      code: "unknown_live_session",
      message: "Realtime session was not found.",
    },
  }), {
    status: 404,
    headers: { "Content-Type": "application/json" },
  });

  try {
    const controller = Object.create(LiveController.prototype);
    Object.assign(controller, {
      sessionId: "expired-session",
      statePollPending: false,
      stateTimer: 42,
      signalClocks: new Map(),
      lastErrorAt: 0,
      media: { stop: async () => { captureStopped = true; } },
      faceTracker: { setTarget: target => assert.equal(target, null) },
      faceStatus,
      speechState,
      onState: state => { renderedState = state; },
      onError: error => errors.push(error.message),
    });

    await controller.refreshState();

    assert.equal(controller.sessionId, null);
    assert.equal(controller.stateTimer, null);
    assert.equal(clearedTimer, 42);
    assert.equal(captureStopped, true);
    assert.equal(renderedState.status, "error");
    assert.equal(renderedState.liveSignals.visionFresh, false);
    assert.equal(faceStatus.textContent, "摄像头画面异常");
    assert.equal(speechState.textContent, "转写不可用");
    assert.equal(errors.length, 1);
    assert.match(errors[0], /实时会话已失效/);
  } finally {
    globalThis.fetch = previousFetch;
    globalThis.window = previousWindow;
  }
});
