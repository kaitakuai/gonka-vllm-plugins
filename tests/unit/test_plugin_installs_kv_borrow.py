# SPDX-License-Identifier: Apache-2.0
"""``register()`` must install the EngineCore KV borrow/return methods.

Without them the availability probe fails, validation drops to the legacy
no-lease layout, and the PoC forward writes its synthetic inputs over KV
blocks the engine is still using. On 1xB300 / DeepSeek-V4-Flash that showed
up as the row at the first sequence boundary drifting between runs while
every other row stayed bit-identical -- a race no measurement could see,
because reading the value synchronised the GPU and hid it.

The install call lives in ``plugin.py`` and nowhere else, so losing it is
silent: every import still resolves, the log line is DEBUG, and the prefill
files stay byte-identical to 0.1.3.
"""
import gonka_poc.plugin as plugin


def test_register_installs_the_borrow_methods(monkeypatch):
    installed = []

    class _Shim:
        @staticmethod
        def install_engine_core_poc_methods():
            installed.append(True)
            return True

    monkeypatch.setattr("gonka_poc._compat.current", lambda: _Shim())
    monkeypatch.setattr(plugin, "_registered", False)
    plugin.register()

    assert installed, (
        "register() no longer installs the KV borrow methods; the PoC forward "
        "will silently share KV blocks with live inference")
