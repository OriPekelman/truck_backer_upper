import * as React from 'react'
import { Group, Line, Circle, RegularPolygon } from 'react-konva'
import { CoordinateSystemTransformation } from './CoordinateSystemTransformation'
import { BoxVisualization } from './BoxVisualization'
import { Trace, TraceEvent, TraceEventType } from '../model/world'
import { Point } from '../math'

interface TraceVisualizationProps {
    traces: Trace[]
    cordSystemTransformer: CoordinateSystemTransformation
    showPaths: boolean
    showOutlines: boolean
    // event markers sit on the object's current position when an episode ends,
    // so they are drawn in a second pass on top of the object itself
    showEvents: boolean
}

interface TraceStyle {
    stroke: string
    dash?: number[]
}

// one style per traced reference point: end of trailer, front of cabin
const traceStyles: TraceStyle[] = [
    { stroke: "#d9534f" },
    { stroke: "#337ab7", dash: [6, 4] }
]

const outlineColor = "#8a8a8a";

interface EventStyle {
    fill: string
    // number of corners of the marker, 0 for a circle
    sides: number
    rotation: number
}

// the clamp and the two terminations end an episode without showing up in its
// terminal error, so each gets a marker of its own
const eventStyles: { [type: number]: EventStyle } = {
    [TraceEventType.JACK_KNIFE]: { fill: "#f0ad4e", sides: 3, rotation: 0 },
    [TraceEventType.HIT_DOCK_WALL]: { fill: "#d9534f", sides: 4, rotation: 0 },
    [TraceEventType.LEFT_AREA]: { fill: "#8e44ad", sides: 4, rotation: 45 },
    [TraceEventType.DOCKED]: { fill: "#5cb85c", sides: 0, rotation: 0 }
};

const eventMarkerRadius = 4.5;
const eventMarkerStroke = "#ffffff";

// trajectories driven earlier are faded out, so the current one stands out
const currentTraceOpacity = 0.85;
const pastTraceOpacity = 0.35;
const currentOutlineOpacity = 0.55;
const pastOutlineOpacity = 0.25;
const currentEventOpacity = 1;
const pastEventOpacity = 0.5;

export class TraceVisualization extends React.Component<TraceVisualizationProps, {}> {

    public constructor(props: TraceVisualizationProps) {
        super(props)
    }

    private mapPath(path: Point[]): number[] {
        let points: number[] = [];
        for (let i = 0; i < path.length; i++) {
            let mapped = this.props.cordSystemTransformer.mapIntoNewCordSystem(path[i]);
            points.push(mapped.x, mapped.y);
        }
        return points;
    }

    private visualizePaths(trace: Trace) {
        let pathVis = [];
        for (let i = 0; i < trace.paths.length; i++) {
            let path = trace.paths[i];
            if (path.length == 0) {
                continue;
            }
            let style = traceStyles[i % traceStyles.length];
            let points = this.mapPath(path);
            pathVis.push(<Group key={i}>
                <Circle x={points[0]} y={points[1]} radius={2} fill={style.stroke} />
                <Line points={points} stroke={style.stroke} strokeWidth={1.5} dash={style.dash} lineCap="round" lineJoin="round" />
            </Group>);
        }
        return pathVis;
    }

    private visualizeOutlines(trace: Trace) {
        let outlineVis = [];
        for (let i = 0; i < trace.outlines.length; i++) {
            let polygons = trace.outlines[i];
            for (let j = 0; j < polygons.length; j++) {
                outlineVis.push(<BoxVisualization key={i + "_" + j} points={polygons[j]} color={outlineColor} cordSystemTransformer={this.props.cordSystemTransformer} />);
            }
        }
        return outlineVis;
    }

    private visualizeEvents(events: TraceEvent[]) {
        let eventVis = [];
        for (let i = 0; i < events.length; i++) {
            let event = events[i];
            let style = eventStyles[event.type];
            if (!style) {
                continue;
            }
            let position = this.props.cordSystemTransformer.mapIntoNewCordSystem(event.position);
            if (style.sides == 0) {
                eventVis.push(<Circle key={i} x={position.x} y={position.y} radius={eventMarkerRadius} fill={style.fill} stroke={eventMarkerStroke} strokeWidth={1} />);
            } else {
                eventVis.push(<RegularPolygon key={i} x={position.x} y={position.y} sides={style.sides} radius={eventMarkerRadius} rotation={style.rotation} fill={style.fill} stroke={eventMarkerStroke} strokeWidth={1} />);
            }
        }
        return eventVis;
    }

    public render() {
        let traceVis = [];
        for (let i = 0; i < this.props.traces.length; i++) {
            let trace = this.props.traces[i];
            let isCurrent = i == this.props.traces.length - 1;
            if (this.props.showOutlines) {
                traceVis.push(<Group key={"outlines" + i} opacity={isCurrent ? currentOutlineOpacity : pastOutlineOpacity}>
                    {this.visualizeOutlines(trace)}
                </Group>);
            }
            if (this.props.showPaths) {
                traceVis.push(<Group key={"paths" + i} opacity={isCurrent ? currentTraceOpacity : pastTraceOpacity}>
                    {this.visualizePaths(trace)}
                </Group>);
            }
            if (this.props.showEvents) {
                traceVis.push(<Group key={"events" + i} opacity={isCurrent ? currentEventOpacity : pastEventOpacity}>
                    {this.visualizeEvents(trace.events)}
                </Group>);
            }
        }
        return <Group>{traceVis}</Group>
    }
}
