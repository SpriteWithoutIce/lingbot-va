"""WebSocket client for policy inference (compatible with wan_va_server)."""

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Policy that talks to lingbot-va server over WebSocket."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                )
                # 不加 ping_interval/close_timeout，以兼容旧版 websockets（无此参数会报错）
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as e:
                logging.info("Still waiting for server... (Error: %s)", e)
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError("Error in inference server:\n%s" % response)
        return unpackb(response)

    def reset(self) -> None:
        pass
