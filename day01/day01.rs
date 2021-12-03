use std::fs::File;
use std::io::{self, BufReader, BufRead};


fn main() -> io::Result<()> {
    let f = File::open("../inputs/d01.txt")?;
    let f: Vec<u16> = BufReader::new(f)
        .lines()
        .map(|l| l
            .unwrap()
            .parse::<u16>()
            .unwrap())
        .collect::<Vec<u16>>();  // more idiomatic way to do this other than sandwiched unwraps?

    let part1: u16 = f.windows(2).map(|v| (v[0] < v[1]) as u16).sum();
    let part2: u16 = f.windows(4).map(|v| (v[0] < v[3]) as u16).sum();
    println!("{} {}", part1, part2);

    Ok(())
}
