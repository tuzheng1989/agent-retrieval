Here's a lobby screen design built around **anticipation**—the emotional core of pre-match waiting. This is a 'countdown to combat' moment, not a loading screen.

## Design Direction

**The one memorable element:** a large countdown timer as the visual anchor—big Sofia Sans numerals that shift from calm white → amber → urgent red as time drains, with a subtle pulse on the final seconds. Everything else supports it.

**Layout:** Two-zone composition. Left side (hero zone): huge timer, 'MATCH FOUND' title, generous negative space. Right side: player roster with race + MMR. Bottom: full-width map preview strip.

```tsx
import styled from 'styled-components'
import { motion, AnimatePresence } from 'motion/react'
import { MaterialIcon } from '../material/icon'
import { color, White, DimTextColor } from '../styles/colors'
import { body1, headline1, overline } from '../styles/typography'

const ANIM_SPRING = { type: 'spring', stiffness: 300, damping: 24 }

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${color('neutral900')};
  padding: 48px 56px;
  gap: 40px;
`

const TopRow = styled.div`
  display: flex;
  flex: 1;
  gap: 64px;
  align-items: center;
`

const TimerZone = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 12px;
`

const Overline = styled.div`
  ${overline}
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: ${color('bluegray300')};
  display: flex;
  align-items: center;
  gap: 8px;
`

const PulseDot = styled(motion.span)`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${color('blue400')};
`

const TimerText = styled(motion.div)<{ $urgent: boolean; $warning: boolean }>`
  ${headline1}
  font-size: 120px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  transition: color 400ms cubic-bezier(0.4, 0, 0.2, 1);
  color: ${p =>
    p.$urgent ? color('red400') : p.$warning ? color('amber400') : White()};
`

const PlayerList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 320px;
`

const PlayerRow = styled(motion.div)<{ $ready: boolean }>`
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 20px;
  background: ${color('neutral800')};
  border-left: 3px solid
    ${p => (p.$ready ? color('blue400') : color('neutral600'))};
`

const RaceBadge = styled.div`
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
`

const MmrBadge = styled.span<{ $high?: boolean }>`
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: ${p => (p.$high ? color('amber400') : DimTextColor())};
`

const MapPreview = styled.div`
  height: 160px;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  background: ${color('neutral800')};
`

const SpawnPoint = styled(motion.div)`
  position: absolute;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: transparent;
  border: 2px solid ${color('blue400')};
  box-shadow: 0 0 12px ${color('blue400')};
`
```

```tsx
export function LobbyScreen({
  players,
  secondsRemaining,
  map,
}: {
  players: { name: string; race: string; mmr: number; ready: boolean }[]
  secondsRemaining: number
  map: { name: string; imageUrl: string; spawns: { x: number; y: number }[] }
}) {
  const urgent = secondsRemaining <= 5
  const warning = secondsRemaining <= 15

  const raceColor = (race: string) =>
    ({ terran: '#4fc3f7', protoss: '#ce93d8', zerg: '#aed581', random: '#ffb74d' }[race] ?? '#90a4ae')

  return (
    <Container>
      <TopRow>
        <TimerZone>
          <Overline>
            <PulseDot animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 2, repeat: Infinity }} />
            Match found · locking in
          </Overline>
          <TimerText
            $urgent={urgent}
            $warning={warning}
            key={secondsRemaining}
            initial={{ scale: urgent ? 1.08 : 1 }}
            animate={{ scale: 1 }}
            transition={ANIM_SPRING}
          >
            0:{String(secondsRemaining).padStart(2, '0')}
          </TimerText>
        </TimerZone>

        <PlayerList>
          {players.map((p, i) => (
            <PlayerRow key={p.name} $ready={p.ready}
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ ...ANIM_SPRING, delay: i * 0.1 }}>
              <RaceBadge style={{ background: `${raceColor(p.race)}22` }}>
                <span style={{ color: raceColor(p.race), fontWeight: 700, textTransform: 'capitalize' }}>
                  {p.race[0]}
                </span>
              </RaceBadge>
              <div>{p.name}</div>
              <MmrBadge $high={p.mmr >= 2000}>{p.mmr} MMR</MmrBadge>
            </PlayerRow>
          ))}
        </PlayerList>
      </TopRow>

      <MapPreview>
        <img src={map.imageUrl} alt={map.name} style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.85 }} />
        {map.spawns.map((s, i) => (
          <SpawnPoint key={i} style={{ left: `${s.x}%`, top: `${s.y}%` }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: [0, 1.4, 1], opacity: 1 }}
            transition={{ ...ANIM_SPRING, delay: 0.6 + i * 0.25 }} />
        ))}
        <div style={{ position: 'absolute', bottom: 12, left: 16, ...overline, textTransform: 'uppercase', letterSpacing: '0.15em' }}>
          {map.name}
        </div>
      </MapPreview>
    </Container>
  )
}
```

## Key design decisions

1. **Timer as hero** — 120px tabular numerals, color-shifting (white → `amber400` → `red400`) following the draft-screen urgency pattern. The per-second `key` change triggers a scale pop; at ≤5s it gets a noticeable 1.08 spring.
2. **Staggered entrance** — players and spawn points animate in sequence (0.1s / 0.25s offsets), giving the lobby a 'roster assembling' rhythm instead of popping in all at once.
3. **Race identity through color** — each race gets a tinted badge at 13% opacity with matching text color, using race-adjacent hues; swap to your exact race color tokens from `colors.ts`.
4. **MMR as quiet flex** — tabular numerals, amber for high ratings, muted otherwise. Confident but not shouting.
5. **Negative space does the work** — the flex-1 timer zone against the fixed-width roster creates tension; nothing is cramped.

**Production notes:** use the actual race icon set rather than letter badges, wire `soundIntegration` like `post-match-dialog.tsx` for the final-5-seconds tick, and replace hardcoded colors with the design-system semantic tokens (`--theme-positive`, race colors from `colors.ts`), which I've approximated inline here.
