import * as React from 'react'
import { Group, Circle } from 'react-konva'
import { Truck } from '../model/truck'
import { CoordinateSystemTransformation } from './CoordinateSystemTransformation'

interface DockReferenceVisualizationProps {
    truck: Truck;
    cordSystemTransformer: CoordinateSystemTransformation;
}

const dockReferenceColor = "#d9534f";

/**
 * Rings the point which has to reach the dock under the conventions in force.
 * The demo stops on one point and grades another, so which one is in force has
 * to be visible rather than implied.
 */
export class DockReferenceVisualization extends React.Component<DockReferenceVisualizationProps, {}> {

    public constructor(props: DockReferenceVisualizationProps) {
        super(props)
    }

    public render() {
        let position = this.props.cordSystemTransformer.mapIntoNewCordSystem(this.props.truck.getDockReferencePoint());
        return <Group>
            <Circle x={position.x} y={position.y} radius={6} stroke={dockReferenceColor} strokeWidth={1.5} />
        </Group>
    }
}
