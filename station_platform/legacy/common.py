from ..util import digest, now, parse_time, stamp

class PocError(Exception):
    def __init__(self, code, message, stage="input", status="BLOCKED"):
        self.code, self.message, self.stage, self.status = code, message, stage, status
        super().__init__(message)
