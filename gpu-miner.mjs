import { parentPort, workerData } from "worker_threads";
import { execSync, spawn } from "child_process";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { keccak_256 } from "@noble/hashes/sha3.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const { challengeHex, difficultyHex } = workerData;

function log(msg) {
  console.log(`[GPU Worker] ${msg}`);
}

// ─── GPU Detection ───────────────────────────────────────
function hasNvidiaGPU() {
  const isWin = process.platform === "win32";
  const nullRedirect = isWin ? " 2>NUL" : " 2>/dev/null";
  const paths = isWin
    ? ["nvidia-smi", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"]
    : ["nvidia-smi"];

  for (const bin of paths) {
    try {
      execSync(`${bin}${nullRedirect}`, { timeout: 3000, stdio: "ignore", windowsHide: true });
      return true;
    } catch {}
  }
  return false;
}

function hasROCmGPU() {
  const isWin = process.platform === "win32";
  const nullRedirect = isWin ? " 2>NUL" : " 2>/dev/null";
  const paths = isWin
    ? ["rocm-smi", "C:\\Program Files\\AMD\\ROCm\\bin\\rocm-smi.exe"]
    : ["rocm-smi"];

  for (const bin of paths) {
    try {
      execSync(`${bin}${nullRedirect}`, { timeout: 3000, stdio: "ignore", windowsHide: true });
      return true;
    } catch {}
  }
  return false;
}

// ─── CPU Fallback (fast, same as worker.mjs) ─────────────
const buf = new Uint8Array(64);
const challenge = Uint8Array.from(
  challengeHex.slice(2).match(/.{2}/g),
  (s) => parseInt(s, 16)
);
buf.set(challenge, 0);

const diffBytes = Uint8Array.from(
  difficultyHex.slice(2).padStart(64, "0").match(/.{2}/g),
  (s) => parseInt(s, 16)
);

function writeNonceBE(nonce) {
  let n = nonce;
  for (let i = 63; i >= 32; i--) {
    buf[i] = Number(n & 0xffn);
    n >>= 8n;
  }
}

function lessThan(a, b) {
  for (let i = 0; i < 32; i++) {
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return false;
}

function cpuMine(startNonce, count) {
  let nonce = startNonce;
  const end = nonce + count;
  while (nonce < end) {
    writeNonceBE(nonce);
    const hash = keccak_256(buf);
    if (lessThan(hash, diffBytes)) {
      return { found: true, nonce: nonce.toString(), hash: Buffer.from(hash).toString("hex") };
    }
    nonce++;
  }
  return { found: false, checked: count.toString() };
}

// ─── GPU Mining via Python + CuPy (NVIDIA) ───────────────
async function gpuMineNvidia() {
  const gpuScript = path.join(__dirname, "gpu-keccak-miner.py");
  const isWin = process.platform === "win32";
  const pythonBin = isWin ? "python" : "python3";

  log(`Python binary: ${pythonBin}`);
  log(`GPU script: ${gpuScript}`);

  // Check if Python GPU script exists
  try {
    const { existsSync } = await import("fs");
    if (!existsSync(gpuScript)) {
      log("GPU script not found");
      return null;
    }
    log("GPU script found");
  } catch (e) {
    log(`Error checking GPU script: ${e.message}`);
    return null;
  }

  // Check if cupy is installed
  log("Checking CuPy installation...");
  try {
    const checkCmd = isWin
      ? `${pythonBin} -c "import cupy; print('cupy ok')"`
      : `${pythonBin} -c "import cupy; print('cupy ok')" 2>/dev/null`;
    const result = execSync(checkCmd, {
      encoding: "utf8",
      timeout: 10000,
      windowsHide: true,
    }).trim();
    log(`CuPy check result: ${result}`);
  } catch (e) {
    log(`CuPy not available: ${e.message}`);
    return null;
  }

  log("CuPy available, starting GPU miner...");

  return new Promise((resolve) => {
    const proc = spawn(pythonBin, [gpuScript, challengeHex, difficultyHex], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      const chunk = data.toString();
      stdout += chunk;
      // Log GPU progress from stderr (progress updates)
    });

    proc.stderr.on("data", (data) => {
      const chunk = data.toString();
      stderr += chunk;
      // Forward GPU progress to main console
      process.stderr.write(chunk);
    });

    proc.on("close", (code) => {
      log(`Python process exited with code ${code}`);
      log(`stdout: ${stdout.slice(0, 500)}`);
      if (stderr) log(`stderr: ${stderr.slice(0, 500)}`);

      if (code === 0 && stdout.includes("found")) {
        try {
          const lines = stdout.trim().split("\n");
          const jsonLine = lines.find((l) => l.startsWith("{"));
          if (jsonLine) {
            const result = JSON.parse(jsonLine);
            if (result.found) {
              resolve(result);
              return;
            }
          }
        } catch (e) {
          log(`Parse error: ${e.message}`);
        }
      }
      resolve(null);
    });

    proc.on("error", (err) => {
      log(`Python process error: ${err.message}`);
      resolve(null);
    });

    // Timeout after 120 seconds
    setTimeout(() => {
      log("GPU mining timeout (120s)");
      proc.kill();
      resolve(null);
    }, 120000);
  });
}

// ─── Main GPU Worker Loop ────────────────────────────────
async function main() {
  const hasGPU = hasNvidiaGPU() || hasROCmGPU();

  if (!hasGPU) {
    log("No GPU detected, running as CPU worker");
    let nonce = BigInt(Math.floor(Math.random() * 2 ** 32));
    const BATCH = 500_000n;

    while (true) {
      const result = cpuMine(nonce, BATCH);
      if (result.found) {
        parentPort.postMessage(result);
        process.exit(0);
      }
      nonce += BATCH;
      parentPort.postMessage({ found: false, checked: BATCH.toString() });
    }
  }

  // Try GPU mining
  log("GPU detected, attempting GPU acceleration...");

  const gpuResult = await gpuMineNvidia();
  if (gpuResult && gpuResult.found) {
    log(`GPU FOUND nonce: ${gpuResult.nonce}`);
    parentPort.postMessage(gpuResult);
    process.exit(0);
  }

  // GPU mining not available or didn't find, fallback to CPU
  log("GPU kernel not available or failed, falling back to CPU");

  let nonce = BigInt(Math.floor(Math.random() * 2 ** 32));
  const BATCH = 500_000n;

  while (true) {
    const result = cpuMine(nonce, BATCH);
    if (result.found) {
      parentPort.postMessage(result);
      process.exit(0);
    }
    nonce += BATCH;
    parentPort.postMessage({ found: false, checked: BATCH.toString() });
  }
}

main().catch((err) => {
  log(`Fatal error: ${err.message}`);
  process.exit(1);
});
