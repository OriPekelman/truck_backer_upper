/**
 * A small --key value / --flag parser, so that the CLI needs no dependencies.
 */
export class Args {
    private values: { [key: string]: string } = {};
    private flags: { [key: string]: boolean } = {};
    private used: { [key: string]: boolean } = {};

    public constructor(argv: string[], private knownFlags: string[] = []) {
        for (let i = 0; i < argv.length; i++) {
            let token = argv[i];
            if (token.substring(0, 2) !== "--") {
                throw new Error("Unexpected argument \"" + token + "\"; options look like --name value");
            }
            let name = token.substring(2);
            let eq = name.indexOf("=");
            if (eq >= 0) {
                this.values[name.substring(0, eq)] = name.substring(eq + 1);
                continue;
            }
            if (this.knownFlags.indexOf(name) >= 0) {
                this.flags[name] = true;
                continue;
            }
            let value = argv[i + 1];
            if (value == undefined || value.substring(0, 2) === "--") {
                throw new Error("Option --" + name + " needs a value");
            }
            this.values[name] = value;
            i++;
        }
    }

    public has(name: string): boolean {
        return this.values[name] != undefined || this.flags[name] === true;
    }

    public flag(name: string): boolean {
        this.used[name] = true;
        return this.flags[name] === true;
    }

    public string(name: string, fallback: string): string {
        this.used[name] = true;
        let value = this.values[name];
        return value == undefined ? fallback : value;
    }

    public choice<T extends string>(name: string, allowed: T[], fallback: T): T {
        let value = this.string(name, fallback);
        for (let i = 0; i < allowed.length; i++) {
            if (value === allowed[i]) {
                return allowed[i];
            }
        }
        throw new Error("Option --" + name + " must be one of " + allowed.join(", ") + ", got \"" + value + "\"");
    }

    public number(name: string, fallback: number): number {
        this.used[name] = true;
        let value = this.values[name];
        if (value == undefined) {
            return fallback;
        }
        let parsed = Number.parseFloat(value);
        if (!isFinite(parsed)) {
            throw new Error("Option --" + name + " must be a number, got \"" + value + "\"");
        }
        return parsed;
    }

    public integer(name: string, fallback: number): number {
        let value = this.number(name, fallback);
        if (Math.round(value) !== value) {
            throw new Error("Option --" + name + " must be a whole number, got " + value);
        }
        return value;
    }

    /** Names given on the command line which nothing asked for. */
    public getUnused(): string[] {
        let unused: string[] = [];
        for (let name in this.values) {
            if (!this.used[name]) {
                unused.push(name);
            }
        }
        for (let name in this.flags) {
            if (!this.used[name]) {
                unused.push(name);
            }
        }
        return unused;
    }
}
