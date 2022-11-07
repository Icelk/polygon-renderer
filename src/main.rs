use std::fmt::{self, Display};
use std::ops::Sub;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}
impl Vec2 {
    pub fn dot_product(self, other: Self) -> f64 {
        let a = self;
        let b = other;
        let axbx = a.x * b.x;
        let aybx = a.y * b.x;
        let axby = a.x * b.y;
        let ayby = a.y * b.y;

        // `TODO`: optimize with FMA operations
        let magnitude = axbx * axbx + axby * axby + aybx * aybx + ayby * ayby;
        let direction = axbx + ayby;
        direction / magnitude.sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}
impl From<(f64, f64)> for Point {
    fn from(v: (f64, f64)) -> Self {
        Self { x: v.0, y: v.1 }
    }
}
impl Sub for Point {
    type Output = Vec2;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
#[derive(Debug, Clone)]
struct Points {
    pub list: Vec<Point>,
}
impl Points {
    pub fn new(v: Vec<impl Into<Point>>) -> Self {
        Self {
            list: v.into_iter().map(Into::into).collect(),
        }
    }
    /// May not contain duplicates
    /// (can be fixed by putting the points in a hashmap (O(n)))
    pub fn sort_using_dot_product(mut self) -> OrderedPoints {
        // O(n)
        let n_th_largest_x = |n: usize, l: &mut [Point]| {
            std_dev::percentile::percentile_by(
                l,
                std_dev::percentile::KthLargest::new(n),
                &mut std_dev::percentile::pivot_fn::rand(),
                &mut |a, b| std_dev::F64OrdHash::f64_cmp(a.x, b.x),
            )
        };

        // O(n)
        let p1 = n_th_largest_x(0, &mut self.list).into_single().unwrap();
        // O(n)
        let p1_idx = self.list.iter().position(|p| *p == p1).unwrap();
        self.list.remove(p1_idx);

        let p2 = n_th_largest_x(0, &mut self.list).into_single().unwrap();
        // O(n)
        let p2_idx = self.list.iter().position(|p| *p == p2).unwrap();
        self.list.remove(p2_idx);

        let (first, second) = if p1.y > p2.y { (p2, p1) } else { (p1, p2) };

        let initial_v = second - first;
        let dot_product_from_initial = |p: Point| initial_v.dot_product(p - first);
        // O(n × log n)
        self.list.sort_unstable_by(|a, b| {
            // O(1)
            dot_product_from_initial(*b).total_cmp(&dot_product_from_initial(*a))
        });
        // for i in &self.list {
        // println!("{}", initial_v.dot_product(*i - first));
        // }
        self.list.insert(0, second);
        self.list.insert(0, first);

        self.assume_ordered()
    }
    pub fn assume_ordered(self) -> OrderedPoints {
        OrderedPoints(self)
    }
}
#[derive(Debug, Clone)]
struct OrderedPoints(Points);
impl OrderedPoints {
    /// O(n)
    #[must_use]
    #[inline(always)]
    pub fn contains(&self, p: Point) -> bool {
        if self.0.list.len() < 2 {
            return false;
        }
        #[inline(always)]
        fn on_left_side(p1: Point, p2: Point, p: Point) -> bool {
            // vertical
            if p1.x == p2.x {
                return p.x < p1.x;
            }
            let f = LineFunction::new(p1, p2);
            // println!("{f}, from {p1:?} to {p2:?}");
            let value = f.evaluate(p.x);
            if p2.x < p1.x {
                p.y <= value
            } else {
                p.y >= value
            }
        }
        for points in self.0.list.windows(2) {
            let p1 = points[0];
            let p2 = points[1];
            if !on_left_side(p1, p2, p) {
                return false;
            }
        }
        if !on_left_side(
            *self.0.list.last().unwrap(),
            *self.0.list.first().unwrap(),
            p,
        ) {
            return false;
        }

        true
    }
}
struct OptimizedEvaluator {
    functions: Vec<(LineFunction, bool)>,
}
impl OptimizedEvaluator {
    /// # Panics
    ///
    /// Panics if `points` has fewer than 2 points.
    pub fn new(points: &OrderedPoints) -> Self {
        let mut fns = Vec::with_capacity(points.0.list.len() + 1);
        for points in points.0.list.windows(2) {
            let p1 = points[0];
            let p2 = points[1];
            fns.push((LineFunction::new(p1, p2), p2.x < p1.x));
        }
        {
            let p1 = *points.0.list.last().unwrap();
            let p2 = *points.0.list.first().unwrap();
            fns.push((LineFunction::new(p1, p2), p2.x < p1.x));
        }
        Self { functions: fns }
    }
    pub fn contains(&self, p: Point) -> bool {
        for (f, p2_is_less) in &self.functions {
            let value = f.evaluate(p.x);
            let b = if *p2_is_less {
                p.y <= value
            } else {
                p.y >= value
            };
            if !b {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct LineFunction {
    slope: f64,
    offset: f64,
}
impl Display for LineFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x + {}", self.slope, self.offset)
    }
}
impl LineFunction {
    /// # Panics
    ///
    /// Panics in debug if `p1.x == p2.x`
    #[inline(always)]
    pub fn new(p1: Point, p2: Point) -> Self {
        debug_assert_ne!(
            p1.x, p2.x,
            "you must handle the case when the two points form a vertical line"
        );

        let slope = (p1.y - p2.y) / (p1.x - p2.x);
        let offset = p1.y - p1.x * slope;
        Self { slope, offset }
    }
    #[inline(always)]
    pub fn evaluate(self, x: f64) -> f64 {
        self.slope * x + self.offset
    }
}

fn render(
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    points: &OrderedPoints,
) -> image::RgbImage {
    let x_dist = x_range.end - x_range.start;
    let y_dist = y_range.end - y_range.start;
    let x_res = 512;
    let y_res = 512;

    let optimized_eval = OptimizedEvaluator::new(points);

    let now = Instant::now();
    let img = image::RgbImage::from_fn(x_res, y_res, |x, y| {
        let y = y_res - y;
        let x = x_range.start + x as f64 / x_res as f64 * x_dist;
        let y = y_range.start + y as f64 / y_res as f64 * y_dist;
        if optimized_eval.contains(Point { x, y }) {
            image::Rgb([255, 255, 255])
        } else {
            image::Rgb([0, 0, 0])
        }
    });
    println!("Took {:?}", now.elapsed());
    img
}
#[allow(clippy::uninit_vec, clippy::cast_ref_to_mut)]
fn render_parallel(
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    points: &OrderedPoints,
) -> image::RgbImage {
    use rayon::prelude::*;
    let x_dist = x_range.end - x_range.start;
    let y_dist = y_range.end - y_range.start;
    let x_res = 512;
    let y_res = 512;

    let optimized_eval = OptimizedEvaluator::new(points);

    // init rayon threadpool
    rayon::spawn(|| {});

    let now = Instant::now();
    let xes: Vec<_> = (0..x_res)
        .into_iter()
        .map(|x| x_range.start + x as f64 / x_res as f64 * x_dist)
        .collect();
    let mut storage = Vec::with_capacity((3 * x_res * y_res) as usize);
    unsafe { storage.set_len(storage.capacity()) };
    let img = image::RgbImage::from_raw(x_res, y_res, storage).unwrap();
    (0..y_res).into_par_iter().for_each(|y| {
        let img = &img;
        let y_coord = y_range.start + y as f64 / y_res as f64 * y_dist;
        let img = unsafe { &mut *(img as *const _ as *mut image::RgbImage) };
        for x in 0..x_res {
            let x_coord = xes[x as usize];
            let pixel = if optimized_eval.contains(Point {
                x: x_coord,
                y: y_coord,
            }) {
                image::Rgb([255, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            };
            img.put_pixel(x, y, pixel)
        }
    });
    println!("Took {:?}", now.elapsed());
    img
}

fn main() {
    let x_range = -5.0..2.5;
    let y_range = -2.0..5.;
    // arbetrary polygon rendering (concave) by only checking if an equation matches if the side
    // "extends" into the realm of the point. All those which can include the point have to have it
    // on the right side.
    let pts = Points::new(vec![
        (2., 4.),
        (2., 0.),
        (-2., -1.),
        (-4., 3.),
        (-5., 0.75),
        (-3.5, -0.5),
        (0.55, 4.2),
        (2.6, 2.1),
    ])
    .sort_using_dot_product();

    // concave
    // let pts =
    // Points::new(vec![(1., 0.), (2., 4.), (-1., 2.), (-4., 3.), (-2., -1.)]).assume_ordered();
    render_parallel(x_range, y_range, &pts)
        .save("out.png")
        .unwrap();
    println!(
        "Pts: {:?}, (0,0) {} (-3,1) {}",
        pts,
        pts.contains((0., 0.).into()),
        pts.contains((-2., 2.).into())
    );
}
