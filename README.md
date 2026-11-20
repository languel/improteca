# improteca

An intelligent virtual dancer for the browser. It can be **taught movements**,
**improvises** with what it knows, and can be **guided** by a prompt, by
real-time interaction (a webcam, a partner), or by a choreographic score.

The name is *improvisation* + *-teca* (a collection, as in *biblioteca*): a
library of movement that the dancer draws on and transforms while it dances.

William Forsythe's *Improvisation Technologies* is one inspiration among many:
the idea that improvisation is a set of operations on the body and space
(inscribing lines, transferring the lead to another body part, reorienting,
dropping a point, recombining a phrase), rather than randomness.

## Status

Design stage. Nothing runs yet. See [docs/DESIGN.md](docs/DESIGN.md) for the
architecture, the plan and what is still unknown.

## Principles

- **Runs in the browser.** WebGPU for the learned motion model, three.js for the
  stage. No server needed for dancing.
- **Learned motion, chosen moves.** A pretrained motion model
  ([NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks)) makes the body
  move like a person between poses. What to dance next is chosen by the
  improviser, from movements the dancer has been taught.
- **Teaching means adding to the repertoire.** Movements come from mocap clips
  and the webcam. The model's weights are not retrained.
- **Every choice is legible.** Each move the dancer makes is recorded with what
  it came from, how it was transformed and why.

## Licences

Code: to be decided by the owner. Third-party models and data keep their own
terms: MotionBricks weights are under the NVIDIA Open Model Agreement, and
NVIDIA's sample BVH clips are for evaluation only, so they are never committed
to this repository.
