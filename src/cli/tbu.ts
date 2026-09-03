import { Args } from './args';
import { runTrain, trainUsage, trainFlags } from './train';
import { runRollout, rolloutUsage, rolloutFlags } from './rollout';

const usage = [
    "tbu -- headless training and rollout for the truck backer upper",
    "",
    trainUsage,
    "",
    rolloutUsage,
    ""
].join("\n");

function main(argv: string[]): number {
    let command = argv[0];
    if (command == undefined || command == "--help" || command == "help") {
        console.log(usage);
        return command == undefined ? 1 : 0;
    }
    try {
        if (command == "train") {
            runTrain(new Args(argv.slice(1), trainFlags));
            return 0;
        }
        if (command == "rollout") {
            runRollout(new Args(argv.slice(1), rolloutFlags));
            return 0;
        }
        console.error("Unknown command \"" + command + "\"");
        console.error("");
        console.error(usage);
        return 1;
    } catch (e) {
        console.error("error: " + (e instanceof Error ? e.message : e));
        return 1;
    }
}

process.exit(main(process.argv.slice(2)));
