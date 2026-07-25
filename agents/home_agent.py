from __future__ import annotations

import httpx

from agents.base_agent import BaseAgent

# NOTE ON GOOGLE HOME MINI CONTROL
# ---------------------------------
# Google does not publish a general local-control API for Google Home Mini.
# The Google Assistant SDK (which allowed sending text commands directly to
# an Assistant device) was deprecated for this kind of use. The practical,
# durable way to control a Home Mini — and to fold it into the same
# tool interface as the rest of your smart home — is via Home Assistant:
#   1. Home Assistant's "Google Assistant" integration lets you expose HA
#      entities *to* Google Assistant.
#   2. For playback, Home Assistant's Google Cast integration talks to the
#      Mini directly (it has Chromecast built in) — you can cast media or
#      announcements and control volume without going through Google's
#      cloud at all.
#   3. For "smart home" actions (lights, switches, scenes), those devices
#      are registered in Home Assistant directly and this agent calls HA's
#      REST API to control them.
#
# So: this agent is a Home Assistant client. That's the integration point.


class HomeAgent(BaseAgent):
    name = "home"
    description = (
        "Controls smart home devices (lights, switches, media, scenes) and "
        "the Google Home Mini's speaker, via Home Assistant."
    )

    def __init__(self, base_url: str, token: str, default_room: str = "living_room") -> None:
        self._base_url = base_url.rstrip("/")
        self._default_room = default_room
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10.0,
        )
        super().__init__()

    def register_tools(self) -> None:
        self.add_tool(
            "turn_on",
            "Turn on a device/entity. Instruction: entity_id, e.g. 'light.living_room'.",
            self.turn_on,
        )
        self.add_tool(
            "turn_off",
            "Turn off a device/entity. Instruction: entity_id.",
            self.turn_off,
        )
        self.add_tool(
            "set_brightness",
            "Set light brightness. Instruction: '<entity_id> <0-255>'.",
            self.set_brightness,
        )
        self.add_tool(
            "get_state",
            "Get current state of an entity. Instruction: entity_id.",
            self.get_state,
        )
        self.add_tool(
            "run_scene",
            "Activate a Home Assistant scene. Instruction: scene entity_id, e.g. 'scene.movie_night'.",
            self.run_scene,
        )
        self.add_tool(
            "announce",
            "Speak a TTS announcement on the Google Home Mini. Instruction: the message text.",
            self.announce,
        )

    async def _call_service(self, domain: str, service: str, payload: dict) -> str:
        resp = await self._client.post(f"/api/services/{domain}/{service}", json=payload)
        resp.raise_for_status()
        return f"OK: {domain}.{service}({payload})"

    async def turn_on(self, instruction: str) -> str:
        entity_id = instruction.strip()
        domain = entity_id.split(".")[0]
        return await self._call_service(domain, "turn_on", {"entity_id": entity_id})

    async def turn_off(self, instruction: str) -> str:
        entity_id = instruction.strip()
        domain = entity_id.split(".")[0]
        return await self._call_service(domain, "turn_off", {"entity_id": entity_id})

    async def set_brightness(self, instruction: str) -> str:
        parts = instruction.strip().split()
        if len(parts) != 2:
            return "set_brightness expects '<entity_id> <0-255>'"
        entity_id, brightness = parts
        return await self._call_service(
            "light", "turn_on", {"entity_id": entity_id, "brightness": int(brightness)}
        )

    async def get_state(self, instruction: str) -> str:
        entity_id = instruction.strip()
        resp = await self._client.get(f"/api/states/{entity_id}")
        resp.raise_for_status()
        data = resp.json()
        return f"{entity_id}: {data.get('state')} ({data.get('attributes', {})})"

    async def run_scene(self, instruction: str) -> str:
        entity_id = instruction.strip()
        return await self._call_service("scene", "turn_on", {"entity_id": entity_id})

    async def announce(self, instruction: str) -> str:
        if "|||" in instruction:
            entity_part, message = instruction.split("|||", 1)
            entity_id = entity_part.replace("entity_id:", "").strip()
        else:
            entity_id = f"media_player.{self._default_room}"
            message = instruction
        return await self._call_service(
            "tts",
            "speak",
            {
                "entity_id": "tts.google_translate_en_com",
                "media_player_entity_id": entity_id,
                "message": message.strip(),
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()