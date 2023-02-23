from math import acos, pi, sin, sqrt, tan
from numbers import Real

eta = pi / 2


def cot(x):
    return 1 / tan(x)


def cot_deg(x):
    return cot(x * eta / 90)


def fmt(double):
    return f"{double:.4f}"


class SmallDeterminant(Exception):
    pass


class v2:
    def __init__(self, x, y):
        assert isinstance(x, Real)
        assert isinstance(y, Real)
        self.x = x
        self.y = y

    def dot(self, other):
        assert isinstance(other, v2)
        return self.x * other.x + self.y * other.y

    def norm_squared(self):
        return self.dot(self)

    def norm(self):
        return sqrt(self.norm_squared())

    def __add__(self, other):
        assert isinstance(other, v2)
        return v2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        assert isinstance(other, v2)
        return v2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        assert isinstance(other, Real)
        return v2(self.x * other, self.y * other)

    def __rmul__(self, other):
        assert isinstance(other, Real)
        return v2(self.x * other, self.y * other)

    def __truediv__(self, other):
        assert isinstance(other, Real)
        return v2(self.x / other, self.y / other)

    def normalized(self):
        n = self.norm()
        return v2(self.x / n, self.y / n)

    def __repr__(self):
        return "(" + fmt(self.x) + ", " + fmt(self.y) + ")"


class v3:
    def __init__(self, x, y, z):
        assert isinstance(x, Real)
        assert isinstance(y, Real)
        assert isinstance(z, Real)
        self.x = x
        self.y = y
        self.z = z

    def dot(self, other):
        assert isinstance(other, v3)
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm_squared(self):
        return self.dot(self)

    def norm(self):
        return sqrt(self.norm_squared())

    def __add__(self, other):
        assert isinstance(other, v3)
        return v3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        assert isinstance(other, v3)
        return v3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        assert isinstance(other, Real)
        return v3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        assert isinstance(other, Real)
        return v3(self.x / other, self.y / other, self.z / other)

    def __rmul__(self, other):
        assert isinstance(other, Real)
        return v3(self.x * other, self.y * other, self.z * other)

    def normalized(self):
        n = self.norm()
        return v3(self.x / n, self.y / n, self.z / n)

    def __repr__(self):
        return "(" + fmt(self.x) + ", " + fmt(self.y) + ", " + fmt(self.z) + ")"

    def cross(self, other):
        return v3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x, 
        )

    def degrees_angle_with(self, other):
        return (90 / eta) * (acos(self.normalized().dot(other.normalized())))

    def drop_x(self):
        return v2(self.y, self.z)

    def drop_y(self):
        return v2(self.x, self.z)

    def drop_z(self):
        return v2(self.x, self.y)


class m22:
    def __init__(self, c1, c2):
        assert isinstance(c1, v2)
        assert isinstance(c2, v2)
        self.a = c1.x
        self.b = c1.y
        self.c = c2.x
        self.d = c2.y

    def det(self):
        return self.a * self.d - self.b * self.c

    def inverse(self):
        D = self.det()
        if abs(D) < 0.001:
            raise SmallDeterminant
        return m22(v2(self.d/D, -self.b/D), v2(-self.c/D, self.a/D))

    def row1(self):
        return v2(self.a, self.c)

    def row2(self):
        return v2(self.b, self.d)

    def col1(self):
        return v2(self.a, self.b)

    def col2(self):
        return v2(self.c, self.d)

    def __mul__(self, other):
        if isinstance(other, m22):
            # self is on the left, other is on the right
            c1 = other.col1()
            c2 = other.col2()
            r1 = self.row1()
            r2 = self.row2()
            return m22(
                v2(c1.dot(r1), c1.dot(r2)),
                v2(c2.dot(r1), c2.dot(r2)),
            )

        elif isinstance(other, v2):
            return v2(self.row1().dot(other), self.row2().dot(other))

        elif isinstance(other, Real):
            return m22(
                self.col1() * other, 
                self.col2() * other,
            )

        else:
            return NotImplemented

    def __neg__(self):
        return self * (-1)

    def __pos__(self):
        return self

    def __repr__(self):
        return fmt(self.a) + " " + fmt(self.c) + "\n" + fmt(self.b) + " " + fmt(self.d)


class m33:
    def __init__(self, c1, c2, c3):
        assert isinstance(c1, v3)
        assert isinstance(c2, v3)
        assert isinstance(c3, v3)

        self.a = c1.x
        self.b = c1.y
        self.c = c1.z

        self.d = c2.x
        self.e = c2.y
        self.f = c2.z

        self.g = c3.x
        self.h = c3.y
        self.i = c3.z

    def a_minor(self):
        return m22(
            self.col2().drop_x(),
            self.col3().drop_x(),
        )

    def b_minor(self):
        return m22(
            self.col2().drop_y(),
            self.col3().drop_y(),
        )

    def c_minor(self):
        return m22(
            self.col2().drop_z(),
            self.col3().drop_z(),
        )

    def d_minor(self):
        return m22(
            self.col1().drop_x(),
            self.col3().drop_x(),
        )

    def e_minor(self):
        return m22(
            self.col1().drop_y(),
            self.col3().drop_y(),
        )

    def f_minor(self):
        return m22(
            self.col1().drop_z(),
            self.col3().drop_z(),
        )

    def g_minor(self):
        return m22(
            self.col1().drop_x(),
            self.col2().drop_x(),
        )

    def h_minor(self):
        return m22(
            self.col1().drop_y(),
            self.col2().drop_y(),
        )

    def i_minor(self):
        return m22(
            self.col1().drop_z(),
            self.col2().drop_z(),
        )

    def det(self):
        return \
            self.a * self.a_minor().det() - \
            self.d * self.d_minor().det() + \
            self.g * self.g_minor().det()

    def row1(self):
        return v3(self.a, self.d, self.g)

    def row2(self):
        return v3(self.b, self.e, self.h)

    def row3(self):
        return v3(self.c, self.f, self.i)

    def col1(self):
        return v3(self.a, self.b, self.c)

    def col2(self):
        return v3(self.d, self.e, self.f)

    def col3(self):
        return v3(self.g, self.h, self.i)

    def transpose(self):
        return m33(self.row1(), self.row2(), self.row3())

    def signed_minors_matrix(self):
        return m33(
            v3(+self.a_minor().det(), -self.b_minor().det(), +self.c_minor().det()),
            v3(-self.d_minor().det(), +self.e_minor().det(), -self.f_minor().det()),
            v3(+self.g_minor().det(), -self.h_minor().det(), +self.i_minor().det()),
        )

    def inverse(self):
        D = self.det()
        if abs(D) < 0.001:
            raise SmallDeterminant
        
        return self.transpose().signed_minors_matrix() / D

    def __truediv__(self, other):
        if isinstance(other, Real):
            return m33(
                self.col1() / other,
                self.col2() / other,
                self.col3() / other,
            )

        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, m33):
            # self is on the left, other is on the right
            r1 = self.row1()
            r2 = self.row2()
            r3 = self.row3()
            c1 = other.col1()
            c2 = other.col2()
            c3 = other.col3()
            return m33(
                v3(c1.dot(r1), c1.dot(r2), c1.dot(r3)),
                v3(c2.dot(r1), c2.dot(r2), c2.dot(r3)),
                v3(c3.dot(r1), c3.dot(r2), c3.dot(r3)),
            )

        elif isinstance(other, v3):
            return v3(
                self.row1().dot(other), 
                self.row2().dot(other),
                self.row3().dot(other),
            )

        elif isinstance(other, Real):
            return m33(
                self.col1() * other,
                self.col2() * other,
                self.col3() * other,
            )

        else:
            return NotImplemented

    def __neg__(self):
        return self * (-1)

    def __pos__(self):
        return self

    def __repr__(self):
        return \
            fmt(self.a) + " " + fmt(self.d) + " " + fmt(self.g) + "\n" + \
            fmt(self.b) + " " + fmt(self.e) + " " + fmt(self.h) + "\n" + \
            fmt(self.c) + " " + fmt(self.f) + " " + fmt(self.i)


class fake_camera_setup:
    camera_pos = v3(0, 0, 0)
    camera_x = v3(1, 0, 0)                                              # you must take care that...
    camera_y = v3(0, 1, 0)                                              # ...these three vectors...
    camera_z = v3(0, 0, 1)                                              # ...are orthonormal, or else incorrect results will be obtained
    camera_frame = m33(camera_x, camera_y, camera_z)                    # camera_x, camera_y, camera_z are columns of the matrix
    device_screen_ax = 45                                               # degrees (half-aperture of camera in x-direction)
    device_screen_hw = 1000                                             # pixels (half-width of phone screen)
    screen_distance = device_screen_hw * cot_deg(device_screen_ax)      # pixels

    @classmethod
    def world_to_camera_space(cls, in_world):
        assert isinstance(in_world, v3)
        return cls.camera_frame.transpose() * (in_world - cls.camera_pos)

    @classmethod
    def camera_space_to_screen(cls, in_camera_space):
        assert isinstance(in_camera_space, v3)
        return (in_camera_space * cls.screen_distance / in_camera_space.z).drop_z()

    @classmethod
    def world_to_screen(cls, in_world):
        in_camera_space = cls.world_to_camera_space(in_world)
        return cls.camera_space_to_screen(in_camera_space)

    @classmethod
    def screen_to_camera_space(cls, on_screen, z):
        assert isinstance(on_screen, v2)
        assert isinstance(z, Real)
        return v3(on_screen.x, on_screen.y, cls.screen_distance) * (z / cls.screen_distance)

    @classmethod
    def camera_space_to_world(cls, in_camera_space):
        assert isinstance(in_camera_space, v3)
        return cls.camera_pos + cls.camera_frame * in_camera_space

    @classmethod
    def screen_to_world(cls, on_screen, z):
        in_camera_space = cls.screen_to_camera_space(on_screen, z)
        return cls.camera_space_to_world(in_camera_space)


def cook_up_example(
    A_position,
    AB_direction,
    other_thing_that_will_be_turned_into_BC_direction,
    AB_length,
    AC_length
):
    unit_vector_from_world_A_to_world_B = AB_direction.normalized()
    unit_vector_from_world_B_to_world_C = \
        other_thing_that_will_be_turned_into_BC_direction.cross(unit_vector_from_world_A_to_world_B).normalized()
    unit_vector_from_world_B_to_world_C = \
        (-1) * unit_vector_from_world_B_to_world_C.cross(unit_vector_from_world_A_to_world_B).normalized()

    world_A = A_position
    world_B = world_A + unit_vector_from_world_A_to_world_B * AB_length
    world_C = world_B + unit_vector_from_world_B_to_world_C * AC_length
    world_D = world_A + unit_vector_from_world_B_to_world_C * AC_length

    screen_A = fake_camera_setup.world_to_screen(world_A)
    screen_B = fake_camera_setup.world_to_screen(world_B)
    screen_C = fake_camera_setup.world_to_screen(world_C)
    screen_D = fake_camera_setup.world_to_screen(world_D)

    return (screen_A, screen_B, screen_C, screen_D), (world_A, world_B, world_C, world_D)


def check_is_square(*args):
    if isinstance(args[0], tuple) or isinstance(args[0], list):
        assert len(args) == 1
        ABCD = args[0]

    else:
        ABCD = args

    assert len(ABCD) == 4
    assert all(isinstance(x, v3) for x in ABCD)

    A = ABCD[0]
    B = ABCD[1]
    C = ABCD[2]
    D = ABCD[3]

    s1 = B - A
    s2 = C - B
    s3 = D - C
    s4 = A - D

    d1 = C - A
    d2 = D - B

    sl1 = s1.norm()
    sl2 = s2.norm()
    sl3 = s3.norm()
    sl4 = s4.norm()

    dl1 = d1.norm()
    dl2 = d2.norm()

    assert abs(sl1 - sl2) < 0.00001
    assert abs(sl2 - sl3) < 0.00001
    assert abs(sl3 - sl4) < 0.00001
    assert abs(sl4 - sl1) < 0.00001
    assert abs(dl1 - dl2) < 0.00001

    assert abs(s1.dot(s2)) < 0.00001
    assert abs(s2.dot(s3)) < 0.00001
    assert abs(s3.dot(s4)) < 0.00001
    assert abs(s4.dot(s1)) < 0.00001
    assert abs(d1.dot(d2)) < 0.00001


def three_corners_main_solver(screen_A, screen_B, screen_D, real_world_sidelength):
    a = fake_camera_setup.screen_to_camera_space(screen_A, 1).normalized()
    b = fake_camera_setup.screen_to_camera_space(screen_B, 1).normalized()
    d = fake_camera_setup.screen_to_camera_space(screen_D, 1).normalized()

    assert isinstance(d, v3)

    def point_d0_on_d_ray_such_that_a_d0_is_perpendicular_to_a_b0(b0):
        assert isinstance(d, v3)
        # equation:
        # ((lambda * d) - a).dot(b0 - a) == 0
        # lambda * d.dot(b0 - a) - a.dot(b0 - a) == 0
        # lambda * d.dot(b0 - a) == a.dot(b0 - a)
        # lambda == a.dot(b0 - a) / d.dot(b0 - a)
        DENOMINATOR = d.dot(b0 - a)
        if abs(DENOMINATOR) < 0.0001:
            raise ZeroDivisionError
        mister_lambda = a.dot(b0 - a) / DENOMINATOR
        return d * mister_lambda

    def norm_squared_ratio_at_b_coefficient(coefficient):
        b0 = b * coefficient

        try:
            d0 = point_d0_on_d_ray_such_that_a_d0_is_perpendicular_to_a_b0(b0)
            return (d0 - a).norm_squared() / (b0 - a).norm_squared()

        except ZeroDivisionError:
            raise

    INITIAL_COEFFICIENT = 0.5
    MAX_COEFFICIENT = 2
    IMPATIENT_INCREASE_MULTIPLIER = 1.05
    SLOW_CRAWL_INCREASE_MULTPLIER = 1.001

    MAX_REVERSALS = 6
    STEPS_PER_REVERSAL = 10
    
    switching_up_pair = None
    switching_down_pair = None
    last_ratio = 0
    next_coefficient = INITIAL_COEFFICIENT

    while next_coefficient < MAX_COEFFICIENT:
        last_coefficient = next_coefficient

        if last_ratio < 0.01:
            next_coefficient = last_coefficient * IMPATIENT_INCREASE_MULTIPLIER

        else:
            next_coefficient = last_coefficient * SLOW_CRAWL_INCREASE_MULTPLIER

        try:
            new_ratio = norm_squared_ratio_at_b_coefficient(next_coefficient)

            if new_ratio >= 1 and last_ratio <= 1:
                    assert switching_up_pair is None
                    assert last_coefficient != INITIAL_COEFFICIENT
                    switching_up_pair = (last_coefficient,  next_coefficient)

            if new_ratio <= 1 and last_ratio >= 1:
                    assert switching_down_pair is None
                    assert last_coefficient != INITIAL_COEFFICIENT
                    switching_down_pair = (last_coefficient,  next_coefficient)
            
            last_ratio = new_ratio

        except ZeroDivisionError:
            print("zero division at", next_coefficient)

    def get_root_for_lower_upper(lower, upper):
        step = (upper - lower) / STEPS_PER_REVERSAL

        num_iterations = 0
        num_reversals = 0
        sign = 1

        x = lower
        x_ratio = norm_squared_ratio_at_b_coefficient(x)

        while num_reversals < MAX_REVERSALS:
            next_x = x + sign * step
            next_x_ratio = norm_squared_ratio_at_b_coefficient(next_x)

            if (1 - x_ratio) * (1 - next_x_ratio) <= 0:
                sign *= -1
                num_reversals += 1
                step /= STEPS_PER_REVERSAL

            x = next_x
            x_ratio = next_x_ratio

            num_iterations += 1

            if num_iterations >= MAX_REVERSALS * STEPS_PER_REVERSAL:
                raise ValueError

        return x
        
    def from_a0_b0_c0_d0_to_ABCD_in_camera_space(a0, b0, c0, d0):
        m = real_world_sidelength / (a0 - b0).norm()
        return a0 * m, b0 * m, c0 * m, d0 * m
        

    def finish_up(winning_coefficient):
        b0 = winning_coefficient * b
        d0 = point_d0_on_d_ray_such_that_a_d0_is_perpendicular_to_a_b0(b0)
        c0 = b0 + (d0 - a)
        A, B, C, D = from_a0_b0_c0_d0_to_ABCD_in_camera_space(a, b0, c0, d0)
        return (
            fake_camera_setup.camera_space_to_world(A),
            fake_camera_setup.camera_space_to_world(B),
            fake_camera_setup.camera_space_to_world(C),
            fake_camera_setup.camera_space_to_world(D),
        )

    solutions = []

    if switching_up_pair is not None:
        up_coefficient = get_root_for_lower_upper(switching_up_pair[0], switching_up_pair[1])
        solutions.append(finish_up(up_coefficient))

    if switching_down_pair is not None:
        dn_coefficient = get_root_for_lower_upper(switching_down_pair[1], switching_down_pair[0])
        solutions.append(finish_up(dn_coefficient))

    return solutions


def check_solution(screen_ABCD, solution_tuple):
    check_is_square(solution_tuple[0], solution_tuple[1], solution_tuple[2], solution_tuple[3])
    print("")
    print("(A projection) compare:", screen_ABCD[0], "with", fake_camera_setup.world_to_screen(solution_tuple[0]))
    print("(B projection) compare:", screen_ABCD[1], "with", fake_camera_setup.world_to_screen(solution_tuple[1]))
    print("(C projection) compare:", screen_ABCD[2], "with", fake_camera_setup.world_to_screen(solution_tuple[2]))
    print("(D projection) compare:", screen_ABCD[3], "with", fake_camera_setup.world_to_screen(solution_tuple[3]))


def announce(string):
    print("########################")
    print(string)
    print("########################")
    print("")


def pretty_print_tuple(solution_tuple, header=None):
    print("")
    if header is not None:
        announce(header + ':')
    print("   A:", solution_tuple[0])
    print("   B:", solution_tuple[1])
    print("   C:", solution_tuple[2])
    print("   D:", solution_tuple[3])


def pretty_print_line(string):
    print("")
    print(string)


def solution_angle_to_vertical(solution_tuple):
    AB = solution_tuple[1] - solution_tuple[0]
    AC = solution_tuple[3] - solution_tuple[0]
    normal_towards_camera_if_A_is_top_left_corner_of_QRcode = AB.cross(AC)
    return normal_towards_camera_if_A_is_top_left_corner_of_QRcode.degrees_angle_with(v3(0, 1, 0))


QR_code_size = 0.02

screen_ABCD, actual_ABCD = cook_up_example(
    v3(0.1, 0.1, 1),
    v3(1, 0, 0),
    v3(0, -1, 0),
    QR_code_size,
    QR_code_size,
)

pretty_print_tuple(screen_ABCD, "screen_ABCD")
pretty_print_tuple(actual_ABCD, "actual_ABCD")

solns = three_corners_main_solver(screen_ABCD[0], screen_ABCD[1], screen_ABCD[3], QR_code_size)

for i, soln in enumerate(solns):
    pretty_print_tuple(soln, f"PRINTOUTS FOR SOLUTION {i + 1}")
    pretty_print_line(f"angle_to_vertical: {solution_angle_to_vertical(soln)}")
    check_solution(screen_ABCD, soln)

print("")