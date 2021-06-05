pub fn Vec2(comptime T: type) type {
    if (!(@typeInfo(T) == .Float or @typeInfo(T) == .Int)) {
        @compileError("You dum dum, you can't do addition over " ++ @typeName(T) ++ "!");
    }

    return extern struct {
        x: T,
        y: T,

        const Self = @This();

        pub fn new(x: T, y: T) Self {
            return Self { .x = x, .y = y };
        }
    };

}

pub fn Vec3(comptime T: type) type {
    if (!(@typeInfo(T) == .Float or @typeInfo(T) == .Int)) {
        @compileError("You dum dum, you can't do addition over " ++ @typeName(T) ++ "!");
    }

    return extern struct {
        x: T,
        y: T,
        z: T,

        const Self = @This();

        pub fn new(x: T, y: T, z: T) Self {
            return Self { .x = x, .y = y, .z = z };
        }
    };
}
