import net from 'node:net';

const [host, portArg] = process.argv.slice(2);
const port = Number(portArg);
const maxAttempts = 60;
const delayMs = 2000;

function tryConnect() {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host, port }, () => {
      socket.end();
      resolve();
    });
    socket.setTimeout(2000);
    socket.on('error', reject);
    socket.on('timeout', () => {
      socket.destroy();
      reject(new Error('timeout'));
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      await tryConnect();
      console.log(`[wait-for] ${host}:${port} listo (intento ${attempt}).`);
      return;
    } catch {
      console.log(`[wait-for] esperando ${host}:${port}... (intento ${attempt}/${maxAttempts})`);
      await sleep(delayMs);
    }
  }
  console.error(`[wait-for] ${host}:${port} no respondió tras ${maxAttempts} intentos.`);
  process.exit(1);
}

main();
