import * as React from 'react'
import { Group, Line, Circle } from 'react-konva'
import { CoordinateSystemTransformation } from './CoordinateSystemTransformation'
import { BoxVisualization } from './BoxVisualization'
import { Trace } from '../model/world'
import { Point } from '../math'

interface TraceVisualizationProps {
    traces: Trace[]
    cordSystemTransformer: CoordinateSystemTransformation
    showPaths: boolean
    showOutlines: boolean
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

// trajectories driven earlier are faded out, so the current one stands out
const currentTraceOpacity = 0.85;
const pastTraceOpacity = 0.35;
const currentOutlineOpacity = 0.55;
const pastOutlineOpacity = 0.25;

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
        }
        return <Group>{traceVis}</Group>
    }
}
