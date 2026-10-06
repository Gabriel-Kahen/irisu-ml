# Original music rules and Spotify mapping

Verified 2026-10-06. This documents normal puzzle mode, the mode implemented by
the website. The original game's story scenes and Metsu mode are separate.

## Selection rules

Normal gameplay does **not** change songs at score thresholds. It retains the
song chosen on the title screen. Clicking the original title cycles the three
normal gameplay songs; the choice is saved. A fresh save selects **Zero
Communication**. Score milestones advance story content, not the running
puzzle's background track.

Normal game over uses **Game Over 1**, independent of score. There is no normal
level-100 exception in the inspected game-over music branch. **Ending Jingle**
is not a verified normal level-100 cue and should not be substituted for it.
The selected gameplay song can resume for a new run.

These conclusions come from the original v2.03 engine, inspected with
`objdump -D -Mintel`, rather than inferred from soundtrack ordering:

- Binary: `reference/game/irisu-v2.03-en/irisu.exe`; SHA-256
  `0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255`.
  As recorded in `manifest.md`, the English patch replaces data archives;
  this executable is the Japanese v2.03 engine.
- `0x4080b8` allocates save data using type metadata at `0x439510`, whose
  initializer points to `0x433940`. The normal BGM index at initializer offset
  `+0x08` is `2`; the Metsu index at `+0x20` is `3`.
- `0x40ec9c` reads the saved music index and dispatches `0..5` to
  `irisu_01..irisu_06`. The normal three are indices `0..2`. Neither score nor
  gameplay level is part of this selection.
- The title click handler at `0x40d262..0x40d272` invokes the switch routine
  `0x40edac`. Normal mode increments the saved index modulo three. Metsu cycles
  indices `3..5`; a later story-state unlock permits all six.
- `0x40adc4..0x40ae2e`, the `ScenePlay` game-over entry, selects `irisu_gos1`
  unconditionally for normal mode. The level-100 branch applies only to Metsu:
  `i_go03B` (Game Over 4), versus `i_go05A` (Game Over 3) below level 100.
- `irisu_gos2` is used by `ScenePhoto` at `0x4092a4`, not selected by a normal
  gameplay score range.

The composer's [original soundtrack notes](https://musmus.main.jp/irisu/)
identify the first three compositions as gameplay music. The ignored extracted
`reference/archives/extracted-v2.03-base/dat/musics.txt` supplies the original
names. No original audio or presentation resource needs redistribution.

## Official Spotify release

The composer's [TuneCore artist page](https://www.tunecore.co.jp/artists/watson?lang=en)
links to the [official release](https://linkco.re/TyEM4HCS?lang=en), released
2026-10-02. Its Spotify link resolves to album
[`72x9BgVuSuOGqmeGgTbAOV`](https://open.spotify.com/album/72x9BgVuSuOGqmeGgTbAOV).
The album has 20 tracks, including original cues, four 2026 rearrangements, and
a bonus track. Use the originals for gameplay.

Track IDs and titles below were checked directly in the public
[Spotify album embed](https://open.spotify.com/embed/album/72x9BgVuSuOGqmeGgTbAOV)
page's `__NEXT_DATA__.props.pageProps.state.data.entity.trackList`:

| Original file | Spotify title | Spotify track ID |
|---|---|---|
| `irisu_01` | Staring at the Ceiling for About Ten Hours | `69Syxg5Ujwse0b2Cbn7zw4` |
| `irisu_02` | I Didn't Talk to Anyone Today, Either | `6eoGq11wAa2DXTs5Qenszg` |
| `irisu_03` | Zero Communication | `69ocfwfs4udHqebi4e2ouO` |
| `irisu_gos1` | Game Over 1 | `4DDJ7CfXUhSDf2Gpp06szh` |

## Playback boundaries

All music must play in a visible Spotify embed. Do not extract Spotify preview
URLs or serve game audio. The [Spotify iFrame API](https://developer.spotify.com/documentation/embeds/references/iframe-api)
supports selecting a track, play/pause, seeking, and playback events. Browser
autoplay restrictions and Spotify account/content availability still apply;
the player may require the listener to press Play and may provide previews.

The website intentionally cycles the first three gameplay tracks automatically,
as requested on 2026-10-06. This is a convenience, not an original score-based
progression rule. Only a loss interrupts that rotation with Game Over 1. The game
loops its music, but a Spotify embed is not a sample-accurate game-audio loop
engine. Any repeat behavior must use supported player controls and respect
pauses; it must not present preview playback as full-track playback. A game-over
cue is a one-shot, not another looping gameplay song.
