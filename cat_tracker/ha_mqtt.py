"""Publish CatTracker state to Home Assistant via MQTT discovery (MQTT integration)."""

from __future__ import annotations

import json
import re
import threading
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None  # type: ignore[assignment]


def _safe_id(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", (s or "main").strip())[:48] or "main"


class HaMqttPublisher:
    """
    Binary sensor (cat in frame) + sensor (identified names) with Home Assistant MQTT discovery.
    """

    def __init__(
        self,
        *,
        broker: str,
        port: int = 1883,
        user: str | None = None,
        password: str | None = None,
        camera_id: str = "main",
        camera_label: str = "CatTracker",
        topic_prefix: str = "cattracker",
        discovery_prefix: str = "homeassistant",
    ) -> None:
        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed (pip install paho-mqtt)")
        self._cid = _safe_id(camera_id)
        self._label = (camera_label or self._cid).strip() or self._cid
        base = topic_prefix.strip().strip("/") or "cattracker"
        self._topic_base = f"{base}/{self._cid}"
        self._discovery = discovery_prefix.strip().strip("/") or "homeassistant"
        self._client = mqtt.Client(client_id=f"ct_{self._cid}"[:22], clean_session=True)
        if user:
            self._client.username_pw_set(user, password or "")
        self._client.on_connect = self._on_connect
        self._broker = broker
        self._port = int(port)
        self._lock = threading.Lock()
        self._discovery_sent = False
        self._last_cat: bool | None = None
        self._last_names: str | None = None

        self._t_binary = f"{self._topic_base}/cat_present"
        self._t_sensor = f"{self._topic_base}/cats_named"

    def _on_connect(self, client: object, userdata: object, flags: object, rc: object, *args: object) -> None:
        try:
            ok = int(rc) == 0
        except (TypeError, ValueError):
            ok = rc == 0
        if ok:
            self._send_discovery()

    def _send_discovery(self) -> None:
        with self._lock:
            if self._discovery_sent:
                return
            dev = {
                "identifiers": [f"cattracker_{self._cid}"],
                "name": f"{self._label}",
                "model": "CatTracker",
                "manufacturer": "CatTracker",
            }
            uid_bin = f"cattracker_{self._cid}_cat_present"
            bin_cfg = {
                "name": f"{self._label} cat present",
                "unique_id": uid_bin,
                "device_class": "occupancy",
                "state_topic": self._t_binary,
                "payload_on": "ON",
                "payload_off": "OFF",
                "device": dev,
            }
            uid_txt = f"cattracker_{self._cid}_cats_named"
            txt_cfg = {
                "name": f"{self._label} cats identified",
                "unique_id": uid_txt,
                "state_topic": self._t_sensor,
                "icon": "mdi:cat",
                "device": dev,
            }
            self._client.publish(
                f"{self._discovery}/binary_sensor/{uid_bin}/config",
                json.dumps(bin_cfg),
                retain=True,
            )
            self._client.publish(
                f"{self._discovery}/sensor/{uid_txt}/config",
                json.dumps(txt_cfg),
                retain=True,
            )
            self._discovery_sent = True

    def start(self) -> bool:
        try:
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.loop_start()
            print(
                f"[ha-mqtt] Home Assistant MQTT → {self._broker}:{self._port} (prefix {self._topic_base})",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[ha-mqtt] could not connect to MQTT broker: {e}", flush=True)
            return False

    def publish(self, cat_in_frame: bool, cats_named: list[str]) -> None:
        try:
            if not self._client.is_connected():
                return
        except AttributeError:
            return
        onoff = "ON" if cat_in_frame else "OFF"
        names = ", ".join(cats_named) if cats_named else ""
        with self._lock:
            if self._last_cat != cat_in_frame:
                self._client.publish(self._t_binary, onoff, retain=True)
                self._last_cat = cat_in_frame
            if self._last_names != names:
                self._client.publish(self._t_sensor, names, retain=True)
                self._last_names = names

    def stop(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
