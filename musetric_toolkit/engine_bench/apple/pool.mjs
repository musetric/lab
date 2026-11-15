import net from 'node:net';

export const startPool = (forwardedPort, serverPort, size = 12) => {
  let idle = 0;
  let closed = false;
  const sockets = new Set();
  const open = () => {
    if (closed) {
      return;
    }
    idle++;
    const phone = net.connect(forwardedPort, '127.0.0.1');
    sockets.add(phone);
    let used = false;
    const retire = () => {
      if (!used) {
        used = true;
        idle--;
        setTimeout(fill, 200);
      }
    };
    phone.once('data', (first) => {
      retire();
      const server = net.connect(serverPort, '127.0.0.1');
      sockets.add(server);
      server.write(first);
      phone.pipe(server);
      server.pipe(phone);
      server.on('error', () => phone.destroy());
      server.on('close', () => {
        sockets.delete(server);
        phone.destroy();
      });
    });
    phone.on('error', () => {});
    phone.on('close', () => {
      sockets.delete(phone);
      retire();
    });
  };
  const fill = () => {
    while (!closed && idle < size) {
      open();
    }
  };
  fill();
  return {
    close: () => {
      closed = true;
      for (const socket of sockets) {
        socket.destroy();
      }
    },
  };
};
