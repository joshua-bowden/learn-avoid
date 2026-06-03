# Auto-Augmenting Robot Manipulation Demonstrations to Learn to Avoid Collisions

**Joshua Bowden, jjosh**
**June 3, 2026**

---

## Background and Setup

Current state-of-the-art robot manipulation demos are very good at completing tasks given that they have an ideal setup for the task. Critical inspection of the robot environment reveals that there are no humans or obsatcles in the way; at most there are distracting objects on the table. As robots move from robot-oriented spaces, like a factory specially designed for them, to human spaces, they need to be aware of and able to avoid collisions with humans and the environment. This work is partly inspired by my day job in automation at a therapeutics company where we are trying to move robots from a dedicated workcell (instruments are lined up for an enclosed arm that memorizes the exact position to operate each one) to the wet lab where there is a dynamic environment, scientists, and $100k instruments.

When we step back from manipulation and look at robotics as a whole, we notice that there is a whole class of robots primarily focused on navigation and obstacle avoidance. Localization and mapping enables turtlebots, robot dogs, drones, and self-driving cars to move in their environment while building an explicit map and understanding what space they can occupy. These robots do this both with traditional methods, such as LIDAR mapping, and with learned methods, like 3d reconstruction and Gaussian splatting. 

Meanwhile, manipulators have a traditional form of obstacle avoidance known as motion planning, where the joints of the arm are checked against a 3D map of the world while trying to go from a given pose to a target pose. However, the field does not seem to have thought about representing this idea in learning-based models. Robotics foundation models are along a spectrum with fully latent understanding on one side, and on the other side are attempts to pull out explicit understanding of things like object recognition, object pose, or grasp pose. But mapping of the environment and the position of the whole arm (joints and end effector) is missing from this spectrum.

To set up the problem, we 


## Approach

Describe relevant background information, prior work, or setup assumptions. Use one or more paragraphs as needed.

## Evaluation and Results

Explain what you did: tools, steps, parameters, and how data or results were collected.

## Conclusion

Summarize outcomes, observations, or metrics. Reference figures or recordings in the sections below.

## References


## Media (GIFs and Videos)

### GIFs (images)

Place GIF files in the repo (e.g. `assets/demo.gif`), then embed with standard Markdown image syntax:

```markdown
![Short description of what the GIF shows](assets/demo.gif)
```

You can also use a full URL:

```markdown
![Demo animation](https://example.com/path/to/demo.gif)
```

On GitHub, you can drag a GIF into the issue/PR comment box or README editor to upload it; GitHub will insert a hosted URL you can paste into the line above.

### Videos

**Option 1 — Link to a file in the repo** (simplest; works everywhere):

```markdown
[Watch the demo video](assets/demo.mp4)
```

