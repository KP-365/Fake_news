# Interface Design

## Design read

- **Surface:** focused application UI for one consequential analysis task.
- **Audience:** students and researchers reviewing a headline or article, primarily on desktop but with a usable mobile path.
- **Single job:** submit text and understand the fixed classifier decision, its uncertainty, the independent NLI context, and the explanation.
- **Risk:** users may mistake retrieved web context for a replacement verdict. The classifier label must remain visually and verbally final.
- **Content:** one long text input, a compact decision record, up to five variable-length evidence links, and a short generated explanation.
- **Platform:** local Gradio web app with keyboard, pointer, and touch input; responsive from 320px to wide desktop.

## Evidence and thesis

The leading archetype is application UI. Linear and Superhuman are useful observations for stable control placement and fast state recognition; Intercom is the contrast because a conversation-first composition would obscure the decision record.

The visual thesis is an editorial evidence desk: warm paper-toned canvas, ink-first typography, ruled sections, and a single cobalt action. The first glance is the fixed label, the second is uncertainty and NLI context, and the third is evidence and explanation. Borders separate provenance roles; shadows and decorative gradients are unnecessary.

## System

- **Type:** Manrope for interface and prose, IBM Plex Mono for numeric signals and process labels.
- **Color roles:** ink for primary content, slate for secondary copy, cobalt for actions and links, red for fake, green for real, amber for context-only NLI.
- **Geometry:** 8px spacing rhythm, 12px control radius, 16px panel radius, shared left edges, 44px minimum action height.
- **Motion:** named color, border, background, and shadow transitions only; reduced motion removes interpolation.
- **Responsive behavior:** two columns on wide screens, one ordered column below 880px, with input before results.

## States and checks

- Empty submission produces a clear validation error and preserves the input.
- Queued analysis exposes Gradio's pending state and permits one model request at a time.
- Success preserves the final label while marking NLI and evidence as context only.
- Missing Anthropic key leaves classifier and verifier results visible with setup guidance.
- Retrieval or NLI failure leaves the classifier label intact and identifies verification as unavailable.
- Evidence URLs are restricted to HTTP(S), escaped, and opened in a new tab.
- Desktop and mobile screenshots, keyboard focus, contrast, 200% text, reduced motion, and long evidence content are the release checks for this page.
