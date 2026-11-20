# Implementing Improvisation Technologies

Goal: the dancer improvises with William Forsythe's *Improvisation
Technologies* (ZKM | Karlsruhe and Forsythe, CD-ROM 1999, online 2023:
<https://improvisation-technologies.zkm.de/>) as **operations on a body**. Each
operation is something the dancer can do, a prompt or score can ask for, the
log can name, and the stage can draw (the CD-ROM's own lines over the video
are the model for the visualisation).

Source: the site's 60 lecture clips and Forsythe's narration (captions, one
file per clip: `captions/<id>.vtt`). Clip numbers are given below as
references. The videos and texts are © ZKM | Karlsruhe and William Forsythe:
we study them, we do not copy them into this repository or the deployed site.

## How the operations fit the engine

The engine already has the right three layers:

1. **Key poses → motion.** MotionBricks fills in whole-body, physically
   plausible motion between 1–4 target poses (0.6–2 s per generation). An
   operation that is a *shape* becomes target poses, made by inverse
   kinematics on the SOMA skeleton.
2. **Corrections on the played motion.** The constraint layer already moves
   limbs with IK every frame (foot planting). An operation that is a *path*
   (inscription) becomes an effector that follows a curve on top of the
   generated motion, the same way.
3. **Phrases.** The repertoire, improviser and composer already transform
   learned phrases (mirror, retrograde, tempo, fragment, accumulate). The
   Reorganizing operations extend this set.

New foundations, needed by almost everything below:

- **Body points**: named points on the body, not only joints: fingertips,
  wrist, elbow, shoulder, hip, knee, heel, toe, head, sternum, pelvis, plus
  points along a bone (a forearm's midpoint) and surfaces of a limb (back of
  the hand, inside of the arm: an orientation as well as a position).
- **Lines and planes in the room or attached to the body**: a line between
  two body points, a line extruded from a point, a line or plane left
  standing in space, each either *in the body's frame* or *in the room's
  frame* (the distinction in Transporting Lines, 122).
- **A whole-body point follower**: IK that puts any body point (hip, elbow,
  head, not only hands and feet) on a target, with the rest of the body
  following (pelvis translation and spine included), within the joint
  limits.
- **A score**: an operation list (`{ op, points, frame, params, duration }`)
  that the improviser, a prompt (through the Brain) or a written score can
  produce, and the log shows by name.

## Lines (1xx)

| Operation | What it is (from the lecture) | Implementation |
| --- | --- | --- |
| Point-Point Line, Imagining Lines (111) | A line between two points (fingertips, or two points of one body part). It can be left standing in space, grabbed, moved, rotated, slid along itself. | Line object between two body points; "leave standing" freezes it in room frame; rotate/slide as transforms; draw it. |
| Extrusion (112) | A line extruded from a point, a body part or the floor; a line on the floor extruded makes a plane, which can be flattened or rotated. | Line grown from a body point along a direction over time; floor lines and planes; extrusion paths become effector targets. |
| Matching (113) | Bringing a limb onto an established line (lines collapse onto each other); then remove the original, or match and rotate. | Target pose by IK aligning a bone (or point-point segment) with a standing line, then rotation about it. |
| Folding (114) | Extending a line by folding at a joint. | A limb section rotated about a hinge so the line continues. |
| Bridging (115) | Showing the line between two points (elbow and hip) by spanning it with another limb, then retreating and leaving it as a new line. | Choose two body points; IK a third limb to span them; leave the line standing; chain match, rotate, bridge. |
| Collapsing Points (116) | Bringing two points on the body together; the many relationships inside a pose (elbows–knees triangle, wrist–wrist line...). | Enumerate point pairs of the current pose; target pose that brings a chosen pair together. |
| Dropping Points (117) | Points collapse toward the floor: lines between a body point (head, hip, knee, elbow) and the floor. | Target that lowers a body point to the floor along its line; viz shows the dropped line. |
| Complex Movements (118) | Extending and extruding chained until the motion is complex but relationships stay clear. | Sequences of the above in the score. |
| Inclination Extension (121) | A body section at an angle has a natural trajectory through space, set by its torsions. | Continue a segment's current direction of travel (its inclination) as the next target path. |
| Transporting Lines (122) | Moving a line and shape either with the body or keeping it fixed to the room while the body moves. | The body-frame / room-frame switch on any line or shape. |
| Dropping Curves (123) | Following a curved motion to its "logical mathematical expansion" until it can no longer develop. | Fit the current effector path (spiral or arc) and extrapolate it, as an effector path, until it leaves the reachable space. |
| Parallel Shear (124) | Two parallel lines (thighs and forearms) fold while keeping their relationship. | Constraint keeping two segments parallel while one moves. |
| Approaches (131–134) | Approaching lines from under, back, top, front, with a chosen surface of a limb; precise angle; torsions. | Limb surface frames (back of hand, inside of arm) in the IK target; forearm/upper-arm twist used deliberately. |
| Avoidance (141–144) | Moving around a line, a volume (cylinder), one's own previous position, or a moving path, so that their presence shows. | Obstacles (lines, cylinders, the ghost of a past pose, a predicted path) that target selection and IK keep clear of. Ties in with the echoes view. |
| In General (151–153) | Also toward the back; also the legs and the floor; simple to complex, shifting modalities. | Biases in the improviser: back space, legs, modality changes. |

## Writing (2xx)

| Operation | What it is | Implementation |
| --- | --- | --- |
| Rotating Inscription (211–214) | Writing with any part of the body (hips, elbows), with more than one limb, shifting the point of inscription quickly; a point writes differently from a length (line). | The point follower with an inscribing point that changes during a path; a length inscribes a surface (two points). Viz: the inscription trace in its colour. |
| Universal Writing (215) | Letters in cursive, and block letters exploded into the room, written with Point Dropping and Point Extrusion. | Letter shapes (stroke fonts) as paths, scaled into the kinesphere or the room. |
| Arc and Axis (216) | Take a movement's path, reinscribe it with another point or line, describing a circular or linear axis through it. | Record a path from one joint, replay it (inverted) with another joint around an axis. |
| Cone vs cylinder (221) | Motion motivated from the end (a cone from the joint) versus internally (a cylinder). | Two modes of the follower: rotate a limb about its base (cone) or translate its base (cylinder, whole body). |
| U-ing and O-ing (222–227) | U or O shapes (also S, triangles) around imaginary axes from joints, internally motivated; can be applied to an existing movement to transform it. | Path generators around joint axes; applied as an additive offset to a phrase's effector paths. |
| Room Writing (231) | Read the room's geometry (a doorknob is a circle) and describe it, knock it out of place. | A small room model (door, window, corners) whose shapes become paths. |
| In General (241–242) | Establish lines by many gestures (wipe, hop, stamp, dot, smear, kick); writing displaces you through space. | Gesture styles as path renderings (dotted, stamped, wiped); travel from writing. |

## Reorganizing (3xx), on learned phrases

| Operation | What it is | Implementation |
| --- | --- | --- |
| Spatial Reorientation (311–314) | The room's front attached to a body part; movements re-placed in the new orientation; a whole combination put on a line, with or without its turns. | Re-express target poses in a frame attached to a chosen joint; "on a line" straightens the root path, keeping or removing turns. |
| Fragmentation (321) | Where each part was, placed in absolute space without the original orientation. | Key poses as absolute body-point positions, reordered and reoriented. |
| Spatial Recovery (322–323) | Go to where body parts were, out of the original order, also backwards in time. | Targets from a phrase's points in shuffled or reversed order (extends our retrograde). |
| Compression (331–333) | The phrase's main vectors drawn by one body part with the rest anchored; in time (abbreviated, folded arms); brushed onto the floor like calligraphy. | Map the phrase's root/effector paths onto one point; time compression as fast, folded variants; floor projection as a foot path. |
| Amplification (334) | A very short section enlarged with fold, open, rotate, extend, match, drop. | Pick a short segment; expand it with a sequence of Lines operations. |
| Adjectival Modification (335) | Apply a quality ("with a bounce") to a phrase. | Our intentions (energy, size, stillness...) applied to a phrase; new qualities (bounce) as modulations. |
| Isometries (341–345) | A form transferred to another body part or scale, keeping its sense, force and relation; shapes to floor patterns. | Shape transfer: a path or angle from one joint re-created, scaled, on another joint, or on the floor as a travel pattern. |

## Additions (4xx)

| Operation | What it is | Implementation |
| --- | --- | --- |
| CZ (421) | Pressure between two limbs, then rotate them as far as possible through the torsions. | Contact constraint between two body points plus twist; later. |
| Soft-Body-Part Trajectory (422–423) | Moving along the surface of the body, pushing and pulling, trajecting the body into space; residual motion elsewhere. | Contact sliding along a body surface, then release; later, needs body surfaces. |

## Order of work

1. **Body points, lines and the score**: the body point set; lines and
   planes (body or room frame); draw them like the CD-ROM overlays; a score
   format and an Operations panel; the log names operations.
2. **Lines operations as target poses**: Point-Point Line, Matching,
   Bridging, Collapsing Points, Dropping Points, Folding, Extrusion,
   Transporting Lines: IK target poses fed to MotionBricks; lines drawn as
   they are made and left standing.
3. **Inscription**: the whole-body point follower on top of generated
   motion; Rotating Inscription with shifting points; U-ing and O-ing; Arc
   and Axis; Dropping Curves; Universal Writing with a stroke font.
4. **Reorganizing** on the repertoire: Reorientation, Fragmentation, Spatial
   Recovery, Compression (space, time, floor), Amplification, Isometries,
   Floor Pattern, Adjectival Modification.
5. **Avoidance**: lines, volumes, own past body (echoes), moving paths.
6. **Additions**: CZ and Soft-Body-Part Trajectory (contact on the body).
7. **Language**: the Brain turns prompts into scores ("bridge elbow to hip,
   leave the line, match it with the left forearm and rotate").

Throughout: Forsythe's principle in 153 (simple to complex, always shifting
between modalities) is the improviser's policy, and every operation is shown
on stage and named in the log.

## Open question

Forsythe's demonstrations would be the best teaching material (pose
estimation of the clips, then the repertoire learns his own lines). The
clips are © ZKM | Karlsruhe and Forsythe: that needs their permission before
it goes beyond private study, and nothing from them is committed here.
