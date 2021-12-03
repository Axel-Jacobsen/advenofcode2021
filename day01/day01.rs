use std::fs::File;
use std::io::{self, BufReader, BufRead};
// use std::cmp::Ordering;


fn quantify<I, T>(data: &I, predicate: Option<fn(T) -> bool>) -> usize
where T: Iterator<Item=T>,
      I: Ord {
    match predicate {
        Some(p) => data.filter(p).count(),
        None => data.filter(|l| l && true).count(),
    }
}


fn main() -> io::Result<()> {
    let f = File::open("input.txt")?;
    let f = BufReader::new(f)
        .lines()
        .map(|l| l
            .unwrap()
            .parse::<u16>()
            .unwrap());  // more idiomatic way to do this other than sandwiched unwraps?

    let part1 = quantify(&f.zip(f.next()), None);
    // let part1 = quantify(f.zip(f, None);
    Ok(())
}
